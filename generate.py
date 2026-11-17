"""加载 checkpoint 直接续写，不需要重新训练。

整个文件只有三步：读 checkpoint → 按里面记的配置重建模型 → 喂 prompt 生成。
存取工具（存 / 读 / 重建）直接复用 train.py，训练逻辑本身不会被执行。

用法：
    python -m transformm.generate                      # 默认 checkpoint + 默认 prompt
    python -m transformm.generate --prompt "Customers buy from" --tokens 80
    python -m transformm.generate --ckpt transformm/checkpoints/last.pt --temperature 0.6 --top-k 20
"""
import os
import sys

# 直接 python xxx.py 时 sys.path[0] 是本目录，`from transformm import` 会失败；把父目录加进来。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch

from transformm.train import (DEFAULT_CKPT, build_model_from_ckpt, generate,
                              load_checkpoint)


def parse_args():
    parser = argparse.ArgumentParser(description="加载 checkpoint 续写文本")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="checkpoint 路径")
    parser.add_argument("--prompt", default="The key to building rapport is")
    parser.add_argument("--tokens", type=int, default=60, help="续写多少 token")
    parser.add_argument("--temperature", type=float, default=0.8, help="<1 更保守，>1 更随机")
    parser.add_argument("--top-k", type=int, default=50, help="只在概率最高的 k 个里采样，0 表示关闭")
    parser.add_argument("--seed", type=int, default=0, help="采样随机种子（同种子结果完全可复现）")
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt = load_checkpoint(args.ckpt)              # 顺带校验架构超参是否与当前一致
    model = build_model_from_ckpt(ckpt)
    model.eval()                                   # 推理：关 dropout，结果才稳定

    n_params = sum(p.numel() for p in model.parameters())
    print(f"checkpoint: {args.ckpt}")
    print(f"step {ckpt['step']} | 参数量 {n_params} | 保存时 loss {ckpt['losses']}")
    print(f"采样: temperature={args.temperature} | top_k={args.top_k} | seed={args.seed}")

    torch.manual_seed(args.seed)
    text = generate(model, args.prompt, new_tokens=args.tokens,
                    temperature=args.temperature, top_k=args.top_k)

    # 同 seed 再生成一次必须逐字相同：generate 内部会 eval()，这里靠它验证
    # 模型确实进了推理模式（dropout 一旦生效，两次结果必然不同）
    torch.manual_seed(args.seed)
    again = generate(model, args.prompt, new_tokens=args.tokens,
                     temperature=args.temperature, top_k=args.top_k)
    assert text == again, "同一 seed 两次生成不一致，检查是否误把模型留在 train() 模式"

    print(f"\nprompt: {args.prompt!r}")
    print(text)


if __name__ == "__main__":
    main()
