"""
 N × TransformerBlock (多个 Transformer 块)
│   ├── LayerNorm
│   ├── Multi-Head Attention
│   ├── LayerNorm
│   └── Feed Forward Network

"""
import os
import sys

# 直接 python xxx.py 时 sys.path[0] 是本目录，`from transformm import` 会失败；把父目录加进来。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn

from transformm import d_model_len, ffn_expansion, num_heads as default_num_heads
from transformm.do_LayerNorm import LayerNorm
from transformm.do_attention import MultiHeadAttention
from transformm.do_ffn import FFN
from transformm.do_pos_encoding import get_pe_eb


class TransformerBlock(nn.Module):
    """Pre-LN 结构的 Transformer 块：LN → 子层 → 残差，重复两次。

    形状契约（B=batch, T=seq_len, C=d_model）：

        x                   [B, T, C]     输入
        LN1(x)              [B, T, C]     先归一化再进注意力（Pre-LN，比 Post-LN 好训）
        mha(...)            [B, T, C]     头内聚合后投影回 d_model
        h1 = x + ...        [B, T, C]     残差 1：注意力的输出直接加进主干
        LN2(h1)             [B, T, C]     在残差主干上再归一化
        ffn(...)            [B, T, C]     逐位置升维到 ffn_expansion*C 再降回 C
        out = h1 + ...      [B, T, C]     残差 2：基准是 h1，不是原始 x

    注意：两次残差的基准不同。第二次若写成 x + ffn(...)，注意力子层的输出就永远进不了
    主干（只剩 FFN 一条路），形状不报错、loss 也能降，属于静默错误。
    """

    def __init__(self, d_model, num_heads: int = default_num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.LN1 = LayerNorm(d_model)
        self.LN2 = LayerNorm(d_model)
        self.ffn = FFN(d_model)
        self.mha = MultiHeadAttention(num_heads=num_heads, d_model_len=d_model)

    def forward(self, x, is_causal: bool = False):
        """两段残差子层。

        Args:
            x: [batch, seq_len, d_model] 形状的张量，最后一维必须等于 d_model。
            is_causal: 透传给 MultiHeadAttention；语言模型训练/推理必须为 True，
                否则位置 i 能看到后面的 token（等于直接看答案）。

        Returns:
            与输入同形的张量 [batch, seq_len, d_model]。
        """
        assert x.dim() == 3, f"TransformerBlock 要求 [batch, seq_len, d_model]，实际 {tuple(x.shape)}"
        assert x.size(-1) == self.d_model, \
            f"最后一维应为 d_model={self.d_model}，实际 {x.size(-1)}"
        h1 = x + self.mha(self.LN1(x), is_causal=is_causal)
        return h1 + self.ffn(self.LN2(h1))


if __name__ == "__main__":
    x = get_pe_eb("""猫吃鱼""")          # [1, 8, 8]：batch=1，seq_len=8（cl100k 按字节切中文）
    batch, seq_len, d_in = x.shape
    assert d_in == d_model_len, f"最后一维应为 d_model_len={d_model_len}，实际 {d_in}"

    blk = TransformerBlock(d_model_len, num_heads=default_num_heads)
    blk.eval()                          # 关掉 FFN 里的 dropout，block 内部才可复现
    output = blk(x)

    # 形状契约：block 不改变任何一个维度
    assert output.shape == x.shape, \
        f"输出应与输入同形 {tuple(x.shape)}，实际 {tuple(output.shape)}"
    # 残差把输入带到了输出，但子层必须真的改变了它，否则等于整块白算
    assert not torch.allclose(output, x), "输出与输入完全相同，子层没有起作用"

    # 逐项复现 block 内部：第二次残差的基准是 h1 而不是 x（写错这里会挂）
    h1 = x + blk.mha(blk.LN1(x))
    manual = h1 + blk.ffn(blk.LN2(h1))
    assert torch.allclose(output, manual, atol=1e-6), \
        f"与手写前向不一致，最大误差 {(output - manual).abs().max().item()}"

    # Pre-LN：送进子层之前先归一化，注意力看到的输入均值≈ 0
    assert blk.LN1(x).mean(-1).abs().max() < 1e-6, "LN1 之后均值不为 0"

    # 回归测试：把 FFN 输出置零，剩下必须是 h1（x + 注意力贡献），不能退化成 x
    with torch.no_grad():
        saved_w = blk.ffn.linear2.weight.clone()
        saved_b = blk.ffn.linear2.bias.clone()
        blk.ffn.linear2.weight.zero_()
        blk.ffn.linear2.bias.zero_()
    out_no_ffn = blk(x)
    with torch.no_grad():
        blk.ffn.linear2.weight.copy_(saved_w)
        blk.ffn.linear2.bias.copy_(saved_b)
    assert torch.allclose(out_no_ffn, h1, atol=1e-6), "FFN 置零后输出应等于 h1"
    assert not torch.allclose(out_no_ffn, x), "注意力子层的贡献丢失在残差里了"

    # 每个子层的参数都必须拿到梯度，说明没有分支被旁路
    # 上面改过 linear2 的权重（in-place），必须重新前向一次才能反传，否则撞 version 报错
    blk.zero_grad()
    (blk(x) * torch.randn(batch, seq_len, d_model_len)).sum().backward()
    no_grad_params = [n for n, p in blk.named_parameters() if p.grad is None]
    assert not no_grad_params, f"这些参数没有梯度: {no_grad_params}"

    print("输入 shape:", x.shape)                         # torch.Size([1, 8, 8])
    print("输出 shape:", output.shape)                    # torch.Size([1, 8, 8])
    print("注意力贡献 h1-x 最大绝对值:", (h1 - x).abs().max().item())   # 非 0
    print("FFN 输出置零后与 h1 最大误差:", (out_no_ffn - h1).abs().max().item())  # ≈ 0
    print("参数量:", sum(p.numel() for p in blk.parameters()))
