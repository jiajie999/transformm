"""训练入口：因果语言模型（next-token prediction）。

一个训练样本就是一条"错位一位"的序列：

    x = ids[i     : i+T]      输入
    y = ids[i + 1 : i+T+1]   标签：y[t] 就是位置 t 的"答案"（下一个 token）

    logits [B, T, V] ──view(-1, V)──> [B*T, V]
    y      [B, T]    ──view(-1)────> [B*T]
    loss = cross_entropy(logits, y)      # 每个位置都在预测自己的下一个 token

因为注意力带因果 mask（is_causal=True），位置 t 只看得到 0..t，预测 t+1 才是"没偷看"的。

用法：
    python -m transformm.train                      # 默认 300 步，每 100 步存一次 checkpoint
    python -m transformm.train --steps 100 --block-size 64
    python -m transformm.train --resume             # 从上次 checkpoint 续训（含优化器状态与 RNG）
    python -m transformm.generate                   # 加载 checkpoint 直接续写，不重新训练
"""
import os
import sys

# 直接 python transformm/xxx.py 时 sys.path[0] 是包目录，`from transformm import` 会失败；把仓库根目录加进来。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from transformm import (d_model_len, max_len, num_blocks, num_heads,
                        pad_token, vocab_size)
from transformm.do_tokenize import decode, encode
from transformm.main import MiniGPT

# 语料：460KB 英文销售教材，约 11 万 token（cl100k_base）
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sales_textbook.txt"
IGNORE_INDEX = pad_token          # pad 位置不参与 loss（本题语料无 pad，留着备用）

CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
DEFAULT_CKPT = CKPT_DIR / "last.pt"


def model_config():
    """写进 checkpoint 的架构超参。加载时靠它确认两边是同一个模型。"""
    return {
        "vocab_size": vocab_size,
        "d_model": d_model_len,
        "num_heads": num_heads,
        "num_blocks": num_blocks,
        "max_len": max_len,
    }


def save_checkpoint(path, model, optimizer, step, args, losses=None):
    """存 model / optimizer / step / config / RNG。

    为什么 optimizer 也要存：AdamW 每步都用 exp_avg / exp_avg_sq 修正更新方向，
    只存权重等于让优化器"冷启动"。实测（20 步热身后续训 20 步、同一批数据）：
    第 1 步 loss 一致，第 20 步差 0.060 —— 差距会随步数继续累积。
    RNG 也要存：训练 batch 是随机抽的，RNG 不接上，续训拿到的就是另一批数据。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),          # 参数 + 位置编码 buffer
        "optimizer": optimizer.state_dict(),  # 动量 + 已训练的步数
        "step": step,
        "config": model_config(),
        "args": vars(args),                   # 只存基础类型，weights_only=True 才能读
        "losses": losses or {},
        "rng_state": torch.get_rng_state(),
    }, path)
    return path


def load_checkpoint(path, map_location="cpu"):
    """读 checkpoint，并校验架构超参与当前 `transformm/__init__.py` 一致。"""
    path = Path(path)
    assert path.exists(), f"找不到 checkpoint {path}，先跑一次 python -m transformm.train"
    # weights_only=True：只允许张量/基础类型，杜绝 pickle 任意代码执行
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    saved, current = ckpt["config"], model_config()
    diff = {k: (saved.get(k), v) for k, v in current.items() if saved.get(k) != v}
    assert not diff, f"checkpoint 与当前配置不一致 (saved, current): {diff}"
    return ckpt


def build_model_from_ckpt(ckpt):
    """按 checkpoint 里记的配置重建模型并载入权重。"""
    cfg = ckpt["config"]
    model = MiniGPT(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                    num_heads=cfg["num_heads"], num_blocks=cfg["num_blocks"],
                    max_len=cfg["max_len"])
    # strict=True（默认）：少 key 或多 key 都报错，避免“静默半加载”
    model.load_state_dict(ckpt["model"])
    return model


def load_tokens(path: Path = DATA_PATH, val_ratio: float = 0.1):
    """读文本 → token id，按比例切成训练集 / 验证集。

    Returns:
        (train_ids, val_ids)，都是 1D 的 long 张量。
    """
    assert path.exists(), f"找不到语料文件 {path}"
    text = path.read_text(encoding="utf-8")
    ids = torch.tensor(encode(text), dtype=torch.long)
    assert ids.numel() > 1000, f"语料太短（{ids.numel()} 个 token），不足以训练"
    split = int(ids.numel() * (1 - val_ratio))
    return ids[:split], ids[split:]


def get_batch(data, batch_size: int, block_size: int):
    """随机切一批 [x, y]，y 是 x 右移一位。

    Returns:
        x: [batch_size, block_size]         输入
        y: [batch_size, block_size]         标签（每个位置的下一个 token）
    """
    # 起点上限要留出 block_size + 1 个位置，否则 y 的切片会越界
    starts = torch.randint(0, data.numel() - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in starts])
    return x, y


def lm_loss(logits, y):
    """交叉熵：把 [B, T, V] 和 [B, T] 压平成 [B*T, V] 和 [B*T]。"""
    assert logits.dim() == 3 and y.dim() == 2, \
        f"logits 应为 [B, T, V]、y 应为 [B, T]，实际 {tuple(logits.shape)} / {tuple(y.shape)}"
    assert logits.size(-1) == vocab_size, f"最后一维应为词表大小 {vocab_size}，实际 {logits.size(-1)}"
    assert logits.shape[:2] == y.shape, f"logits/y 的 [B, T] 不一致: {tuple(logits.shape[:2])} vs {tuple(y.shape)}"
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


@torch.no_grad()
def estimate_loss(model, data, batch_size: int, block_size: int, batches: int):
    """在若干随机 batch 上取平均 loss（模型未训练时 ≈ ln(V)）。

    评估会抽随机 batch，也就消费全局 RNG；这里进出各存/恢复一次，
    让“评估”对训练过程是**无副作用**的 —— 否则续训的 batch 序列会因评估次数不同而错位。
    """
    rng_state = torch.get_rng_state()
    model.eval()                                   # 关掉 dropout，评估结果才稳定
    total = 0.0
    for _ in range(batches):
        x, y = get_batch(data, batch_size, block_size)
        total += lm_loss(model(x), y).item()
    model.train()
    torch.set_rng_state(rng_state)
    return total / batches


@torch.no_grad()
def generate(model, prompt: str, new_tokens: int = 40, temperature: float = 0.8, top_k: int = 50):
    """自回归生成：每次只取最后一个位置的分布，采样一个 token 接到后面。

    有了因果 mask 才能这样"一步步续写"：前面位置的结果不会被后接的 token 改动。
    没有 KV cache，每步都重算整段（教学实现，够用就行）。
    """
    was_training = model.training
    model.eval()                                   # 推理模式：关 dropout
    ids = torch.tensor([encode(prompt)], dtype=torch.long)
    for _ in range(new_tokens):
        window = ids[:, -max_len:]                 # 位置编码只有 max_len 个位置可用
        logits = model(window)[:, -1, :] / temperature
        if top_k:                                  # 只在概率最高的 top_k 里采样，避免胡言乱语
            kth = torch.topk(logits, top_k).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        ids = torch.cat([ids, nxt], dim=1)
    if was_training:                               # 只恢复调用前的状态，不要无条件 train()
        model.train()
    return decode(ids[0].tolist())


def parse_args():
    parser = argparse.ArgumentParser(description="训练迷你 GPT（next-token prediction）")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=32, help=f"每条样本的 token 数，必须 < max_len={max_len}")
    parser.add_argument("--lr", type=float, default=3e-3, help="d_model 只有 8，lr 太小 300 步降不下来，实测 3e-3 合适")
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--gen-tokens", type=int, default=40, help="训练结束后生成多少 token，0 表示不生成")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="checkpoint 路径（存与续训共用）")
    parser.add_argument("--save-interval", type=int, default=100, help="每多少步存一次，0 表示不存")
    parser.add_argument("--resume", action="store_true", help="从 --ckpt 续训")
    parser.add_argument("--min-improvement", type=float, default=0.5,
                        help="训练后 loss 至少要比 ln(V) 低这么多，否则报错；短跑测试可设 0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    assert 0 < args.block_size < max_len, f"block_size 应在 (0, {max_len}) 内，实际 {args.block_size}"
    torch.manual_seed(args.seed)

    train_data, val_data = load_tokens()
    print(f"语料: {DATA_PATH.name} | train {train_data.numel()} tokens | val {val_data.numel()} tokens")

    model = MiniGPT(vocab_size=vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    # AdamW = Adam + 解耦权重衰减，GPT 系列训练的常用默认
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    # 随机初始化时，每个位置在 V 个候选里基本是均匀乱猜，交叉熵期望就是 ln(V)
    random_guess = math.log(vocab_size)
    print(f"参数量 {n_params} | 随机初始化期望 loss ≈ ln({vocab_size}) = {random_guess:.3f}")

    start_step = 0
    last_eval = None
    if args.resume:
        ckpt = load_checkpoint(args.ckpt)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])   # 动量接上，续训不会“重新热机”
        torch.set_rng_state(ckpt["rng_state"])          # batch 序列也接上
        start_step = ckpt["step"]
        last_eval = ckpt["losses"] or None
        print(f"从 {args.ckpt} 续训: step {start_step} | 保存时 loss {ckpt['losses']}")
        # load_state_dict 会把 param_groups 里的 lr 一并盖回旧值，命令行 --lr 会静默失效
        saved_lr = ckpt["optimizer"]["param_groups"][0]["lr"]
        if saved_lr != args.lr:
            print(f"  注意: checkpoint 里的 lr={saved_lr}，本次 --lr={args.lr}，以本次为准")
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    assert start_step < args.steps, \
        f"checkpoint 已在 step {start_step}，--steps 要比它大才能继续训（否则换个 --ckpt）"

    t0 = time.time()
    for step in range(start_step + 1, args.steps + 1):
        x, y = get_batch(train_data, args.batch_size, args.block_size)

        logits = model(x)                          # is_causal 默认 True：位置 t 看不到 t+1
        loss = lm_loss(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # 梯度裁剪：小模型 + 高 lr 时偶发梯度爆炸，裁到范数 1.0 更稳
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 第一步、每个评估间隔、最后一步都评估一次（保证断言有数可用）
        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            train_loss = estimate_loss(model, train_data, args.batch_size, args.block_size, args.eval_batches)
            val_loss = estimate_loss(model, val_data, args.batch_size, args.block_size, args.eval_batches)
            last_eval = {"train": train_loss, "val": val_loss}
            print(f"step {step:4d} | step loss {loss.item():7.3f} | "
                  f"train {train_loss:7.3f} | val {val_loss:7.3f} | {time.time() - t0:5.1f}s")

        # 定期落盘：中途被杀掉最多损失 save_interval 步
        if args.save_interval and step % args.save_interval == 0 and step != args.steps:
            save_checkpoint(args.ckpt, model, optimizer, step, args, last_eval or {})
            print(f"  └ 已存 checkpoint: {args.ckpt} (step {step})")

    assert last_eval is not None, "没有做过评估，检查 eval_interval 与 steps"
    # 训练结束再存一次：这里存的一定是最终权重
    save_checkpoint(args.ckpt, model, optimizer, args.steps, args, last_eval)
    print(f"最终 checkpoint: {args.ckpt} (step {args.steps})")

    train_loss, val_loss = last_eval["train"], last_eval["val"]
    # 必须真的学到了东西：loss 应明显低于"均匀乱猜"
    assert train_loss < random_guess - args.min_improvement, \
        f"训练后 train loss {train_loss:.3f} 没有明显低于随机猜测 {random_guess:.3f}"
    assert val_loss < random_guess - args.min_improvement, \
        f"训练后 val loss {val_loss:.3f} 没有明显低于随机猜测 {random_guess:.3f}"
    print(f"最终 train {train_loss:.3f} | val {val_loss:.3f} | "
          f"随机猜测 {random_guess:.3f} | 提升 {random_guess - val_loss:.3f} nats")

    if args.gen_tokens > 0:
        prompt = "The key to building rapport is"
        print(f"\n生成（prompt = {prompt!r}，模型很小，内容只能算随机续写）:")
        print(generate(model, prompt, new_tokens=args.gen_tokens))


if __name__ == "__main__":
    main()
