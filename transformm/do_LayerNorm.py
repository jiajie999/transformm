import os
import sys

# 直接 python transformm/xxx.py 时 sys.path[0] 是包目录，`from transformm import` 会失败；把仓库根目录加进来。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn

from transformm import d_model_len, ln_eps
from transformm.do_pos_encoding import get_pe_eb


class LayerNorm(nn.Module):
    """层归一化：对每个 token 自己的 d_model 维归一化，再仿射变换。

        LN(x) = γ * x_norm + β
        x_norm = (x - mean) / sqrt(var + eps)

    形状契约（B=batch, T=seq_len, C=d_model）：

        x                  [B, T, C]        输入，最后一维恒为特征维
        mean(-1)           [B, T, 1]        每个 token 单独统计，跨 batch、跨位置互不影响
        var (-1)           [B, T, 1]        总体方差（unbiased=False），即除以 C 而不是 C-1
        x_norm             [B, T, C]        均值 0、标准差 1
        γ, β               [C]              广播到 [B, T, C]，每个特征维一组可学习参数

    注意：
    - 归一化维度是最后一维（特征维），不是 batch 维：同一个样本内部各 token 各算各的，
      所以与 BatchNorm 不同，不依赖 batch 大小，推理时不需要滑动统计量。
    - 方差必须是总体方差：x.std() 默认 unbiased=True（除以 C-1），会整体放大 sqrt(C/(C-1))。
    - eps 加在方差上再开方（sqrt(var + eps)），不是加在 std 上；不加的话整行相同值时
      std=0 会除出 nan，而 nan 会沿计算图传染整个网络。
    """

    def __init__(self, d_model=d_model_len, eps=ln_eps):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        """计算层归一化。

        Args:
            x: [..., seq_len, d_model] 形状的张量，最后一维必须等于 d_model。
               2D 为单条序列 [seq_len, d_model]，3D 为一批 [batch, seq_len, d_model]。

        Returns:
            与输入同形的张量 [..., seq_len, d_model]。
        """
        assert x.size(-1) == self.d_model, \
            f"最后一维应为 d_model={self.d_model}，实际 {x.size(-1)}"
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


if __name__ == "__main__":
    x = get_pe_eb("""猫吃鱼""")          # [1, 8, 8]：batch=1，seq_len=8（cl100k 按字节切中文）
    batch, seq_len, d_in = x.shape
    assert d_in == d_model_len, f"最后一维应为 d_model_len={d_model_len}，实际 {d_in}"

    ln = LayerNorm(d_model_len)
    output = ln(x)

    # 形状契约：逐位置归一化，不改 batch / seq_len，输出与输入同形
    assert output.shape == x.shape, \
        f"输出应与输入同形 {tuple(x.shape)}，实际 {tuple(output.shape)}"
    # γ=1、β=0 时归一化结果是标准化的：均值 0、方差 1
    assert torch.allclose(output.mean(-1), torch.zeros(batch, seq_len), atol=1e-6), \
        f"每个 token 的均值应为 0，实际最大 {output.mean(-1).abs().max().item()}"
    assert torch.allclose(output.var(-1, unbiased=False), torch.ones(batch, seq_len), atol=1e-6), \
        f"每个 token 的方差应为 1，实际最差 {output.var(-1, unbiased=False).sub(1).abs().max().item()}"

    # 与官方实现对拍：eps 取同一值才能逐元素对齐（nn.LayerNorm 默认 1e-5）
    ref = nn.LayerNorm(d_model_len, eps=ln.eps)
    assert torch.allclose(output, ref(x), atol=1e-5), \
        f"与 nn.LayerNorm 不一致，最大误差 {(output - ref(x)).abs().max().item()}"

    # 整行取值相同（std=0）时必须有限：这正是 eps 存在的理由
    const = torch.full((1, 2, d_model_len), 7.0)
    assert torch.isfinite(ln(const)).all(), "常数行不应出现 nan/inf"

    print("输入 shape:", x.shape)                # torch.Size([1, 8, 8])
    print("输出 shape:", output.shape)           # torch.Size([1, 8, 8])
    print("输出 mean(-1) 最大绝对值:", output.mean(-1).abs().max().item())  # ≈ 0
    print("输出 var(-1) 与 1 的最大偏差:", output.var(-1, unbiased=False).sub(1).abs().max().item())  # ≈ 2e-6
    print("与 nn.LayerNorm 最大误差:", (output - ref(x)).abs().max().item())  # ≈ 2e-7
    print("常数行输出:", ln(const).flatten()[:2], "-> 有限，未出现 nan")  # [0., 0.]
