"""位置编码（Positional Encoding）。

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

位置编码是常量表，不是可学习参数：用 register_buffer 存，能跟着 state_dict / .to(device) 走，
但不会被优化器更新。它的作用是注入"第几个 token"的信息——注意力本身对位置是无感的，
打乱输入顺序，输出只会跟着一起打乱、数值不变。
"""
import os
import sys

# 直接 python transformm/xxx.py 时 sys.path[0] 是包目录，`from transformm import` 会失败；把仓库根目录加进来。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn

from transformm import d_model_len, max_len
from transformm.do_embedding import embedding_build


def pos_encode(d_model=None, seq_max_len=None):
    """生成正弦位置编码表。

    Args:
        d_model: 特征维，不传时取全局 d_model_len（保持旧的无参调用方式可用）。
        seq_max_len: 位置数，不传时取全局 max_len。

    Returns:
        [seq_max_len, d_model] 的常量表，第 pos 行是第 pos 个位置的编码。
    """
    d_model = d_model_len if d_model is None else d_model
    seq_max_len = max_len if seq_max_len is None else seq_max_len
    assert d_model % 2 == 0, f"d_model 必须是偶数（sin/cos 成对写入），实际 {d_model}"

    pos_encoding = torch.zeros(seq_max_len, d_model)
    # PE(pos,2i) = sin(pos/10000^(2i/d_model))
    # PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    for pos in range(seq_max_len):
        for i in range(0, d_model, 2):
            angle = torch.tensor(pos / (10000 ** (i / d_model)))
            pos_encoding[pos, i] = torch.sin(angle)
            pos_encoding[pos, i + 1] = torch.cos(angle)
    return pos_encoding


class PositionalEncoding(nn.Module):
    """把正弦位置编码加到输入上（通常是 embedding 的输出）。

    形状契约（B=batch, T=seq_len, C=d_model）：

        x                 [B, T, C]      输入
        pe                [seq_max_len, C]   常量表，buffer：进 state_dict，但不进优化器
        pe[:T]            [T, C]         只取前 T 行，多余的预计算不用
        .unsqueeze(0)     [1, T, C]      广播到每个 batch 样本（stride=0，不复制数据）
        out               [B, T, C]      与输入同形
    """

    def __init__(self, d_model=d_model_len, seq_max_len=max_len):
        super().__init__()
        self.d_model = d_model
        self.seq_max_len = seq_max_len
        # 用 buffer 而不是 Parameter：常量表不应该被梯度更新
        self.register_buffer("pe", pos_encode(d_model=d_model, seq_max_len=seq_max_len))

    def forward(self, x):
        """给每个位置加上对应的位置编码。

        Args:
            x: [batch, seq_len, d_model] 形状的张量，最后一维必须等于 d_model。

        Returns:
            与输入同形的张量 [batch, seq_len, d_model]。
        """
        assert x.dim() == 3, f"PositionalEncoding 要求 [batch, seq_len, d_model]，实际 {tuple(x.shape)}"
        assert x.size(-1) == self.d_model, \
            f"最后一维应为 d_model={self.d_model}，实际 {x.size(-1)}"
        seq_len = x.size(-2)
        assert seq_len <= self.seq_max_len, \
            f"序列长度 {seq_len} 超过位置编码上限 {self.seq_max_len}"
        return x + self.pe[:seq_len].unsqueeze(0)


def get_pe_eb(text="""cat eat fish
                               fish eat cat"""):
    """文本 → token embedding + 位置编码。

    Args:
        text: 原始文本，按行切分成 batch，每行是一条序列（右侧用 pad_token 补齐）。

    Returns:
        [batch, seq_len, d_model] 的张量。

    仅用于学习/调试：把"切词 + 查 embedding 表 + 加位置编码"一次跑通，方便观察形状。
    生产路径不要用它——embedding_build 每次调用都新建一张随机表，那张表既不可训练、
    也存不进 state_dict；模型里应该写成 pos_encoding(nn.Embedding(...)(token_ids))，
    也就是 main.py 中 MiniGPT 的 forward。
    """
    emb = embedding_build(text)                              # [B, T, C]
    return PositionalEncoding(d_model=emb.size(-1))(emb)     # [B, T, C]


if __name__ == "__main__":
    text = "猫吃鱼"
    x = get_pe_eb(text)
    assert x.shape == (1, 8, d_model_len), \
        f"get_pe_eb 应返回 {(1, 8, d_model_len)}，实际 {tuple(x.shape)}"

    pe = PositionalEncoding(d_model_len)          # 单独取出常量表来检查
    assert pe.pe.shape == (max_len, d_model_len), \
        f"pe 应为 {(max_len, d_model_len)}，实际 {tuple(pe.pe.shape)}"

    # 位置 0：sin(0)=0、cos(0)=1 -> 偶数维全 0，奇数维全 1
    assert torch.allclose(pe.pe[0, 0::2], torch.zeros(d_model_len // 2)), "位置 0 的 sin 部分应为 0"
    assert torch.allclose(pe.pe[0, 1::2], torch.ones(d_model_len // 2)), "位置 0 的 cos 部分应为 1"
    # 不同位置的编码必须不同，否则位置信息等于没加
    assert not torch.allclose(pe.pe[1], pe.pe[2]), "位置 1 和 2 的编码相同"

    # 验证 forward 做的是"加法"：同一份 embedding 加上 PE 后，减回去应还原出 PE
    emb = embedding_build(text)
    assert torch.allclose(PositionalEncoding(d_model_len)(emb) - emb,
                          pe.pe[:emb.size(1)].unsqueeze(0), atol=1e-6), "forward 不是简单的加法"

    # buffer 语义：进 state_dict，但不进 parameters（优化器看不到）
    assert "pe" in pe.state_dict(), "pe 应该在 state_dict 里"
    assert len(list(pe.parameters())) == 0, "pe 不应该出现在 parameters 里"

    print("get_pe_eb 输出 shape:", x.shape)                 # torch.Size([1, 8, 8])
    print("PE 表 shape:", pe.pe.shape)                      # torch.Size([109, 8])
    print("位置 0 前 4 维:", pe.pe[0, :4].tolist())          # [0.0, 1.0, 0.0, 1.0]
    print("位置 1 前 4 维:", pe.pe[1, :4].tolist())          # sin/cos(1) 的逐维结果
    print("参数个数:", len(list(pe.parameters())), "| state_dict 键:", list(pe.state_dict()))
