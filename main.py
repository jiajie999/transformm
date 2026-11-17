"""Transformer 主流程：embedding + 位置编码 → N × TransformerBlock → LayerNorm → LM head。

    token_ids                       [B, T]              整数 id，cl100k_base 切词结果
    embedding(C)                    [B, T, C]           nn.Embedding：V × C 的可学习查找表，是模型参数
    pos encoding                    [B, T, C]           正弦常量表，buffer：进 state_dict 但不训练
    block × num_blocks（串联）        [B, T, C]           每层的输出是下一层的输入，形状始终不变
    final LayerNorm                 [B, T, C]           Pre-LN 结构下，出口处要再归一化一次
    LM head: Linear(C → V)          [B, T, V]           logits，V = vocab_size（cl100k_base 词表，100277）
    softmax(logits[:, -1, :])       [B, V]              推理时只用最后一个位置，得到下一个 token 的分布

注意：forward 默认 is_causal=True，每个位置只能看到自己和左边的 token（GPT 的做法）。
真正训练请看 train.py；这里只做前向形状/流程演示。
"""
import os
import sys

# 直接 python xxx.py 时 sys.path[0] 是本目录，`from transformm import` 会失败；把父目录加进来。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn

from transformm import d_model_len, max_len, num_blocks, num_heads, pad_token, vocab_size
from transformm.do_LayerNorm import LayerNorm
from transformm.do_pos_encoding import PositionalEncoding
from transformm.do_tokenize import encode, encoding
from transformm.transformerBlock import TransformerBlock


class MiniGPT(nn.Module):
    """迷你 GPT：只做前向，不包含 mask 和 weight tying。

    形状契约（B=batch, T=seq_len, C=d_model, V=vocab_size）：

        token_ids          [B, T]           必须是整数张量（dtype=torch.long）
        embedding          [B, T, C]        nn.Embedding 就是一张 V × C 的可学习查找表
        pos encoding       [B, T, C]        PositionalEncoding 只做加法，形状不变
        blocks × N         [B, T, C]        串联，逐层细化（每层注意力带因果 mask）
        ln_final           [B, T, C]
        lm_head            [B, T, V]        最后投影到词表维，才是语言模型需要的输出

    参数归属：embedding / lm_head / blocks / ln_final 都是模型参数（进 parameters() 和
    state_dict()，能被训练和保存）；位置编码表是常量，只进 state_dict 不进 parameters()。
    """

    def __init__(self, vocab_size: int, d_model: int = d_model_len,
                 num_heads: int = num_heads, num_blocks: int = num_blocks,
                 max_len: int = max_len):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # 词表大小由 tokenizer 决定，不是本模块自己拍的数字：传错会在查表时越界
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token)
        # 位置编码是固定公式算出来的常量表，交给 buffer 管理，不进优化器
        self.pos_encoding = PositionalEncoding(d_model=d_model, seq_max_len=max_len)

        # ModuleList 才会把子模块注册进本模块：
        # 用普通 list 的话 parameters() / state_dict() / .to(device) 全都看不到它们
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model=d_model, num_heads=num_heads) for _ in range(num_blocks)]
        )
        self.ln_final = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids, is_causal: bool = True):
        """token id → 每个位置上对下一个 token 的 logits。

        Args:
            token_ids: [batch, seq_len] 整数 id，取值 [0, vocab_size)。
            is_causal: 是否启用因果 mask。默认 True（GPT 语义）；
                仅当想看“全可见”的对照结果时才传 False。

        Returns:
            [batch, seq_len, vocab_size] 的 logits（未过 softmax）。
        """
        assert token_ids.dim() == 2, \
            f"token_ids 应为 [batch, seq_len]，实际 {tuple(token_ids.shape)}"
        assert not torch.is_floating_point(token_ids), \
            f"token_ids 必须是整数，实际 {token_ids.dtype}"
        assert token_ids.numel() == 0 or token_ids.max().item() < self.vocab_size, \
            f"token id 超出词表范围: 最大 {int(token_ids.max().item())} >= vocab_size={self.vocab_size}"

        x = self.embedding(token_ids)                        # [B, T] -> [B, T, C]，查表
        x = self.pos_encoding(x)                             # [B, T, C] -> [B, T, C]，加位置信号

        for blk in self.blocks:                              # 串联：上一层输出喂给下一层
            x = blk(x, is_causal=is_causal)                   # [B, T, C]

        x = self.ln_final(x)                                 # [B, T, C]
        return self.lm_head(x)                               # [B, T, V]


if __name__ == "__main__":
    torch.manual_seed(42)

    # 全局常量 vocab_size 必须与 tokenizer 的实际词表一致，否则 embedding 会越界
    assert vocab_size == encoding.n_vocab, \
        f"vocab_size={vocab_size} 与 tokenizer 词表 {encoding.n_vocab} 不一致"

    model = MiniGPT(vocab_size=vocab_size)
    model.eval()                       # 关掉 FFN 里的 dropout，前向结果才可复现

    token_ids = torch.tensor([encode("猫吃鱼")])           # [1, 8]
    with torch.no_grad():              # 纯推理：不建计算图，省显存也更快
        logits = model(token_ids)

    batch, seq_len = 1, 8              # "猫吃鱼" 被 cl100k_base 切成 8 个 token
    # 形状契约：logits 的最后一维必须是词表大小，而不是 d_model
    assert logits.shape == (batch, seq_len, vocab_size), \
        f"logits 应为 {(batch, seq_len, vocab_size)}，实际 {tuple(logits.shape)}"

    # 可复现：embedding 现在是模型里固定的一份表，同样输入两次结果必须相同
    # （之前 embedding 每次调用都新建随机表，这条断言会挂）
    with torch.no_grad():
        assert torch.equal(model(token_ids), logits), "同一输入的两次前向结果不同"

    # 因果性：改掉最后一个 token，前面位置的 logits 必须一字不变
    # （mask 没接上的话，这条会挂在“未来影响了历史”上）
    token_ids2 = token_ids.clone()
    token_ids2[:, -1] = (token_ids2[:, -1] + 1) % vocab_size
    with torch.no_grad():
        logits2 = model(token_ids2)
    assert not torch.allclose(logits2, logits), "换了最后一个 token，输出却没变"
    assert torch.allclose(logits2[:, :-1], logits[:, :-1], atol=1e-5), \
        f"未来位置影响了历史位置，最大误差 {(logits2[:, :-1] - logits[:, :-1]).abs().max().item()}"

    # 推理接生成时只看最后一个位置，softmax 后是下一个 token 的概率分布
    next_token_probs = torch.softmax(logits[:, -1, :], dim=-1)   # [B, V]
    assert torch.allclose(next_token_probs.sum(-1), torch.ones(1), atol=1e-6), \
        "概率分布之和应为 1"

    # 参数归属：embedding 必须在 parameters() 里（能被训练），位置编码只在 state_dict 里
    params = dict(model.named_parameters())
    assert "embedding.weight" in params, "embedding 没被注册成参数"
    assert params["embedding.weight"].shape == (vocab_size, d_model_len)
    assert "pos_encoding.pe" in model.state_dict(), "位置编码应在 state_dict 里"
    assert "pos_encoding.pe" not in params, "位置编码是常量，不该出现在 parameters() 里"

    # 所有子层都必须被注册进模型，否则 optimizer / state_dict 拿不到它们
    assert len(model.blocks) == num_blocks
    assert all(any(key.startswith(f"blocks.{i}.") for key in model.state_dict())
               for i in range(num_blocks)), "有 block 没被注册进 state_dict"

    # 端到端可训练：反向传播能走到每一个参数，包括 embedding
    model.zero_grad()
    model(token_ids).mean().backward()
    no_grad_params = [name for name, p in model.named_parameters() if p.grad is None]
    assert not no_grad_params, f"这些参数没拿到梯度: {no_grad_params}"
    emb_grad = model.embedding.weight.grad
    assert emb_grad.abs().sum() > 0, "embedding 的梯度全为 0"
    assert emb_grad[pad_token].abs().sum() == 0, "padding_idx 那一行不该有梯度"

    top5 = torch.topk(next_token_probs[0], k=5)
    total = sum(p.numel() for p in model.parameters())
    print("输入 token_ids:", token_ids.shape)                     # torch.Size([1, 8])
    print("logits shape:", logits.shape)                          # torch.Size([1, 8, 100277])
    print("下一个 token 概率分布 shape:", next_token_probs.shape)  # torch.Size([1, 100277])
    print("top-5 token id:", top5.indices.tolist(), "（模型未训练，内容无意义）")
    print("top-5 概率:", [round(p, 6) for p in top5.values.tolist()])
    print("参数量:", total,
          "= embedding", model.embedding.weight.numel(),
          "+ lm_head", model.lm_head.weight.numel(),
          "+ blocks", sum(p.numel() for p in model.blocks.parameters()),
          "+ ln_final", sum(p.numel() for p in model.ln_final.parameters()))
    print("state_dict 里含位置编码:", "pos_encoding.pe" in model.state_dict())
    print("embedding 梯度范数:", round(emb_grad.norm().item(), 6))
