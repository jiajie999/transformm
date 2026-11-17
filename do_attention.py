"""自注意力（Self-Attention）与多头注意力（Multi-Head Attention）。

    Attention = softmax(Q @ K^T / sqrt(d_model)) @ V
    其中 Q = Wq(x), K = Wk(x), V = Wv(x)

- 缩放因子：单头累加 d_model 项，用 sqrt(d_model)；多头每个头只累加 d_k 项，用 sqrt(d_k)。
  不缩放会让 logits 过大、softmax 趋于 one-hot。
- 投影只作用于最后一维（nn.Linear 特性），因此与 seq_len 无关，整句共享同一组权重。
- 输入输出同形 [..., seq_len, d_model]：最后一个维度恒为 d_model，倒数第二个维度才是“序列”。

本文件两个类：
- Attention：单头版本，逐项对应上面的公式，用于建立直觉。
- MultiHeadAttention：把 d_model 切成 num_heads 份并行计算，再 concat 回来过一层 W_o。
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

from transformm import d_model_len, max_len
from transformm.do_pos_encoding import get_pe_eb


class Attention(nn.Module):
    """单头自注意力层：只在 seq 维之间做加权聚合，各 batch 样本互不影响。"""

    def __init__(self, d_model_len):
        super().__init__()
        self.Wk = nn.Linear(d_model_len, d_model_len, bias=False)
        self.Wq = nn.Linear(d_model_len, d_model_len, bias=False)
        self.Wv = nn.Linear(d_model_len, d_model_len, bias=False)
        self.d_model_len = d_model_len

    def forward(self, x):
        """计算自注意力输出。

        Args:
            x: [..., seq_len, d_model] 形状的张量，最后一维必须等于 d_model_len。
               2D 为单条序列 [seq_len, d_model]，3D 为一批 [batch, seq_len, d_model]。

        Returns:
            与输入同形的张量 [..., seq_len, d_model]。
            内部中间结果：[..., seq_len, d_model] -> QK^T -> [..., seq_len, seq_len]
            （softmax 沿最后一维归一化，即每个 query 对所有 key 的权重和为 1）。
        """
        assert x.size(-1) == self.d_model_len, \
            f"最后一维应为 d_model_len={self.d_model_len}，实际 {x.size(-1)}"

        Q = self.Wq(x)
        V = self.Wv(x)
        K = self.Wk(x)

        attention_out = F.softmax(Q @ K.transpose(-2, -1) / self.d_model_len ** 0.5, dim=-1) @ V
        return attention_out


class MultiHeadAttention(nn.Module):
    """多头自注意力层：头与头之间是并行计算，输出是所有头的 concat + 一个全连接层。

    形状契约（B=batch, T=seq_len, H=num_heads, d_k=d_model_len//H）：

        x                       [B, T, d_model]     输入，只支持 3D
        Wq(x)/Wk(x)/Wv(x)       [B, T, d_model]     整句共享一组投影权重，与 seq_len 无关
        .view(B, T, H, d_k)     [B, T, H, d_k]      拆头：d_model 拆成 (H, d_k)
        .transpose(1, 2)        [B, H, T, d_k]      头挪进 batch 维，才能用 @ 批量算
        Q @ K.transpose(-2,-1)  [B, H, T, T]        每个头自己的打分矩阵
        masked_fill(上三角)      [B, H, T, T]        is_causal=True 时把 j > i 的位置置成 -inf
        softmax(dim=-1)         [B, H, T, T]        每个 query 对所有 key 的权重和为 1
        @ V                     [B, H, T, d_k]      按权重聚合 value
        .transpose(1, 2)        [B, T, H, d_k]      头回到最后一维
        .view(B, T, d_model)    [B, T, d_model]     concat 所有头（先 contiguous，否则 view 会报错）
        W_o(...)                [B, T, d_model]     跨头混合，输出与输入同形

    返回：默认只返回 [B, T, d_model]；return_attn=True 时额外返回权重 [B, H, T, T]。

    注意：缩放的除数是 sqrt(d_k)（每个头只累加 d_k 项），不是 sqrt(d_model)。

    因果 mask（is_causal=True）：位置 i 只能看 j <= i 的 key。做法是先 softmax 再
    masked_fill 是错的（权重和不再为 1），必须先把上三角置成 -inf 再 softmax。
    语言模型训练/推理都必须开它，否则模型直接看到答案，loss 会掉得毫无意义。
    """

    def __init__(self, num_heads: int, d_model_len):
        super().__init__()
        self.num_heads = num_heads
        assert d_model_len % num_heads == 0, f"d_model_len={d_model_len} 不能被 num_heads={num_heads} 整除"
        self.d_k = d_model_len // self.num_heads
        self.d_model_len = d_model_len

        self.Wk = nn.Linear(self.d_model_len, self.d_model_len, bias=False)
        self.Wq = nn.Linear(self.d_model_len, self.d_model_len, bias=False)
        self.Wv = nn.Linear(self.d_model_len, self.d_model_len, bias=False)
        self.Wo = nn.Linear(self.d_model_len, self.d_model_len, bias=False)

        # 上三角（不含对角线）为 True 的布尔掩码，按需切片使用。
        # persistent=False：它是常量，不需要进 state_dict，加载旧权重不会缺 key。
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, x, return_attn: bool = False, is_causal: bool = False):
        """计算多头自注意力输出。

        Args:
            x: [batch, seq_len, d_model] 形状的张量，最后一维必须等于 d_model_len。
            return_attn: 为 True 时额外返回注意力权重，用于观察/调试；
                默认 False，保持 [B, T, d_model] -> [B, T, d_model] 的形状契约。
            is_causal: 为 True 时启用因果 mask，位置 i 只能看到 j <= i（语言模型必须开）。

        Returns:
            return_attn=False：输出 [B, T, d_model]，与输入同形。
            return_attn=True：(输出 [B, T, d_model], 权重 [B, H, T, T])，
            权重每一行（每个 query 对所有 key）之和为 1；is_causal=True 时权重矩阵是下三角。
        """
        assert x.dim() == 3, f"MultiHeadAttention 要求 [batch, seq_len, d_model]，实际 {tuple(x.shape)}"
        batch, seq_len, d_in = x.shape
        assert d_in == self.d_model_len, \
            f"最后一维应为 d_model_len={self.d_model_len}，实际 {d_in}"
        Q = self.Wq(x)
        V = self.Wv(x)
        K = self.Wk(x)

        Q = Q.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / self.d_k ** 0.5        # [B, H, T, T]
        if is_causal:
            # 序列长于预分配的掩码时（理论上被位置编码挡住，这里留个兜底）
            if seq_len > self.causal_mask.size(0):
                self.causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device), diagonal=1)
            # 先把未来位置压成 -inf 再 softmax：exp(-inf)=0，权重自然为 0
            scores = scores.masked_fill(self.causal_mask[:seq_len, :seq_len], float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_out = attention_weights @ V
        attention_out = attention_out.transpose(1, 2)
        attention_out = attention_out.contiguous().view(batch, seq_len, self.d_model_len)
        attention_out = self.Wo(attention_out)
        if return_attn:
            return attention_out, attention_weights
        return attention_out


if __name__ == "__main__":
    # seq_len = 3
    # batch = 4
    # num_heads = 2
    # d_model_len = 8
    #
    # x = torch.randn(seq_len, d_model_len)
    # A = Attention(d_model_len)
    # print(A(x))

    # 多头：输入必须是 3D [batch, seq_len, d_model]，且 num_heads 要能整除 d_model_len
    # x_mh = torch.randn(batch, seq_len, d_model_len)
    # M = MultiHeadAttention(num_heads, d_model_len)
    # out_mh = M(x_mh)
    # print(out_mh)
    num_heads = 2
    d_model = d_model_len

    torch.manual_seed(42)
    # 只有一行文本 → batch=1；cl100k_base 按字节切中文，"猫吃鱼" 切出 8 个 token → seq_len=8
    x = get_pe_eb("""猫吃鱼""")
    assert x.dim() == 3, f"MultiHeadAttention 需要 [batch, seq_len, d_model]，实际 {tuple(x.shape)}"
    batch, seq_len, _ = x.shape

    mha = MultiHeadAttention(num_heads, d_model)
    output, attn_weights = mha(x, return_attn=True)

    # 形状契约：输出必须与输入同形 [B, T, d_model]
    assert output.shape == x.shape, \
        f"输出应与输入同形 {tuple(x.shape)}，实际 {tuple(output.shape)}"
    # 每个头一张 [T, T] 打分矩阵
    assert attn_weights.shape == (batch, num_heads, seq_len, seq_len), \
        f"权重形状应为 {(batch, num_heads, seq_len, seq_len)}，实际 {tuple(attn_weights.shape)}"
    # softmax 沿最后一维归一化：每个 query 对所有 key 的权重和为 1（dim 写反这条就会挂）
    row_sums = attn_weights.sum(-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
        "权重每一行之和应为 1"

    # 因果 mask：同一份权重下，位置 i 只能看 j <= i
    mha_causal = MultiHeadAttention(num_heads, d_model)
    mha_causal.load_state_dict(mha.state_dict())      # 沿用同一份权重，差异只来自 mask
    assert len(mha_causal.state_dict()) == len(mha.state_dict()), \
        "causal_mask 不该出现在 state_dict 里（persistent=False）"
    output_c, attn_c = mha_causal(x, return_attn=True, is_causal=True)

    # 上三角必须精确为 0：exp(-inf) 是 0，不是“很小”
    i, j = torch.triu_indices(seq_len, seq_len, offset=1)
    assert attn_c[..., i, j].abs().max() == 0, "上三角注意力权重应为 0"
    assert torch.equal(attn_c, attn_c.tril()), "权重矩阵不是下三角"
    # 加了 mask 之后，每行（每个 query）的权重和仍然是 1
    causal_row_sums = attn_c.sum(-1)
    assert torch.allclose(causal_row_sums, torch.ones_like(causal_row_sums), atol=1e-6), \
        "加 mask 后每行权重和仍应为 1"
    assert not torch.allclose(attn_c, attn_weights), "加 mask 后权重没变，mask 没生效"
    assert not torch.allclose(output_c, output), "加 mask 后输出没变"

    # 因果性：只改最后一个位置的输入，前面位置的输出必须一点不变
    x2 = x.clone()
    x2[:, -1] = x2[:, -1] + 1.0
    out2 = mha_causal(x2, is_causal=True)
    assert not torch.allclose(out2, output_c), "改动最后一个位置没有任何影响，输入没变？"
    assert torch.allclose(out2[:, :-1], output_c[:, :-1], atol=1e-6), \
        f"未来位置影响了历史位置，最大误差 {(out2[:, :-1] - output_c[:, :-1]).abs().max().item()}"

    print("输入 shape:", x.shape)                      # torch.Size([1, 8, 8])
    print("输出 shape:", output.shape)                 # torch.Size([1, 8, 8])
    print("d_k:", mha.d_k, "| 权重 shape:", attn_weights.shape)  # 4 | torch.Size([1, 2, 8, 8])
    print("权重每行之和 (min / max):", row_sums.min().item(), row_sums.max().item())  # 1.0 1.0
    print("因果 mask 第 0 行:", attn_c[0, 0, 0].tolist())        # [1.0, 0, 0, 0, ...] 只能看自己
    print("因果 mask 第 3 行前 5 个:", [round(v, 4) for v in attn_c[0, 0, 3, :5].tolist()])  # 4 个非零



