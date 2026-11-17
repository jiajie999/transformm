import os
import sys

# 直接 python transformm/xxx.py 时 sys.path[0] 是包目录，`from transformm import` 会失败；把仓库根目录加进来。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn

from transformm import d_model_len, dropout_rate, ffn_expansion
from transformm.do_pos_encoding import get_pe_eb


class FFN(nn.Module):
    """逐位置前馈网络：先升维到 ffn_expansion*d_model，过 ReLU，再降回 d_model。

    形状契约（B=batch, T=seq_len）：

        x                    [B, T, d_model]
        linear1              [B, T, ffn_expansion*d_model]   升维，只作用于最后一维，位置之间互不影响
        ReLU                 [B, T, ffn_expansion*d_model]   非线性，否则两层线性叠起来仍是一个线性变换
        linear2              [B, T, d_model]                 降回 d_model，才能与残差相加
        dropout              [B, T, d_model]                 训练时随机置零，eval 时直通

    注意：升维-降维是为了给每个位置更大的非线性变换容量，参数量（约 2*ffn_expansion*d_model^2）
    占整个 Transformer block 的大头；ffn_expansion 取 4（见 transformm/__init__.py）是实践惯例。
    """

    def __init__(self, d_model=None):
        super().__init__()
        # 不传时回退到全局 d_model_len，保持 FFN() 的旧调用方式可用
        self.d_model = d_model if d_model is not None else d_model_len
        self.hidden = self.d_model * ffn_expansion
        self.linear1 = nn.Linear(self.d_model, self.hidden)
        self.linear2 = nn.Linear(self.hidden, self.d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        """逐位置前馈。

        Args:
            x: [..., seq_len, d_model] 形状的张量，最后一维必须等于 d_model。

        Returns:
            与输入同形的张量 [..., seq_len, d_model]。
        """
        assert x.size(-1) == self.d_model, \
            f"最后一维应为 d_model={self.d_model}，实际 {x.size(-1)}"

        x = self.linear1(x)
        x = self.ReLU(x)
        x = self.linear2(x)
        return self.dropout(x)


if __name__=="__main__":
    x = get_pe_eb("""猫吃鱼""")          # [1, 8, 8]：batch=1，seq_len=8（cl100k 按字节切中文）
    batch, seq_len, d_in = x.shape
    assert d_in == d_model_len, f"最后一维应为 d_model_len={d_model_len}，实际 {d_in}"

    ffn = FFN()
    ffn.eval()                          # 关掉 dropout，结果才可复现、才能和手写前向对照
    output = ffn(x)

    # 形状契约：逐位置运算，不改 batch / seq_len，输出与输入同形
    assert output.shape == x.shape, \
        f"输出应与输入同形 {tuple(x.shape)}，实际 {tuple(output.shape)}"

    # 中间层升维到 ffn_expansion*d_model，这也是 FFN 参数量的来源
    hidden = ffn.ReLU(ffn.linear1(x))
    assert hidden.shape == (batch, seq_len, d_model_len * ffn_expansion), \
        f"中间层应为 {(batch, seq_len, d_model_len * ffn_expansion)}，实际 {tuple(hidden.shape)}"

    # eval 模式下 dropout 直通，输出必须等于手工串起来的结果（顺序写反、漏掉 ReLU 都会挂这里）
    manual = ffn.linear2(hidden)
    assert torch.allclose(output, manual, atol=1e-6), "输出与手写前向不一致"

    print("输入 shape:", x.shape)                       # torch.Size([1, 8, 8])
    print("中间层 shape:", hidden.shape)                 # torch.Size([1, 8, 32])
    print("输出 shape:", output.shape)                   # torch.Size([1, 8, 8])
    print("参数量:", sum(p.numel() for p in ffn.parameters()))  # 8*32+32 + 32*8+8 = 552