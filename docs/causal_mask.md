# 因果 mask：为什么必须先把上三角压成 -inf，再 softmax

> 起因：给 MiniGPT 加训练代码时发现，位置 t 的注意力能看到 t+1 及之后的 token ——
> 而训练标签正是「下一个 token」，等于把答案直接喂给模型。
> 结论：`torch.triu(ones, diagonal=1)` 生成的布尔上三角要先用 `masked_fill(-inf)` **再** `softmax`。
> 顺序写反、`diagonal` 写错、或者干脆忘了开 mask，三种都是**不报错的静默错误**。

## 1. 那个布尔矩阵长什么样

```python
self.register_buffer(
    "causal_mask",
    torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1),
    persistent=False,
)
```

实测（`max_len=109`）：

```
shape: (109, 109) | dtype: torch.bool | 字节数: 11881
前 5×5（True=屏蔽，False=可见）:
    [0, 1, 1, 1, 1]
    [0, 0, 1, 1, 1]
    [0, 0, 0, 1, 1]
    [0, 0, 0, 0, 1]
    [0, 0, 0, 0, 0]
```

拆开看每一步：

| 表达式 | 结果 |
|--------|------|
| `torch.ones(109, 109, dtype=torch.bool)` | 全 `True`（bool 的 1，不是 1.0） |
| `torch.triu(..., diagonal=1)` | **保留**第 1 条上对角线以上的部分，其余清零 |
| 最终语义 | `True` = 屏蔽，`False` = 可见 |

`diagonal` 决定从哪条对角线开始保留：

```
diagonal=0:  [[1,1,1,1],        diagonal=1:  [[0,1,1,1],
              [0,1,1,1],                     [0,0,1,1],
              [0,0,1,1],                     [0,0,0,1],
              [0,0,0,1]]                     [0,0,0,0]]
```

## 2. 三步链条

```python
scores = Q @ K.transpose(-2, -1) / self.d_k ** 0.5        # [B,H,T,T]
scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))
attention_weights = F.softmax(scores, dim=-1)
```

以第 0 行为例（`T=4`）：

```
masked_fill 之后: [[1.0, -inf, -inf, -inf]]
再 softmax:       [[1.0, 0.0, 0.0, 0.0]]
```

`exp(-inf) = 0` 是**精确的 0**，不是「很小的正数」。所以上三角的权重断言可以用
`== 0` 而不是 `allclose(0, atol=...)`：

```python
i, j = torch.triu_indices(seq_len, seq_len, offset=1)
assert attn_weights[..., i, j].abs().max() == 0, "上三角注意力权重应为 0"
```

## 3. 为什么不能先 softmax 再 mask

先归一化再置零，每行的权重和就只剩「可见部分占的比例」：

```
T=8, 第 0 行：softmax 后是 8 个 1/8 = 0.125，mask 掉 7 个 → 行和 0.125
```

等于这个 query 的注意力总量丢了 87.5%，`@ V` 之后输出被整体缩小，
而这一层后面还有残差和 LayerNorm，**不会报错，只是模型学得莫名其妙**。

正确语义必须是「**归一化之前**就把不可见的位置排除在分母之外」，这样剩下的可见位置权重和重新归一到 1。
实测加了 mask 之后行和仍然是 `1.0`（`do_attention.py` 里就有一条断言锁这个）。

## 4. `diagonal=0` 会得到 nan

如果把主对角线也 mask 掉，位置 i 连**自己**都看不到，第 0 行整行 `-inf`：

```
softmax([-inf, -inf, -inf]) → [nan, nan, nan]
```

`diagonal=1` 保证了位置 i 至少能看到自己，第 i 行永远有 i+1 个可见 key，
分母不会为 0 —— 这是「必须留对角线」的唯一理由。

## 5. 怎么验证 mask 真的接上了

### 5.1 权重层面（单元级，`do_attention.py`）

```python
output_c, attn_c = mha(x, return_attn=True, is_causal=True)
assert attn_c[..., i, j].abs().max() == 0                      # 上三角精确为 0
assert torch.equal(attn_c, attn_c.tril())                      # 是下三角
assert torch.allclose(attn_c.sum(-1), torch.ones_like(...))    # 行和仍为 1
assert not torch.allclose(attn_c, attn_weights)                # 确实变了
```

输出：

```
因果 mask 第 0 行: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
因果 mask 第 3 行前 5 个: [0.2238, 0.1714, 0.1278, 0.477, 0.0]   ← 只有 4 个非零
```

### 5.2 端到端层面（`main.py`）

最有力的判据是「**改未来，前文不能动**」：

```python
token_ids2 = token_ids.clone()
token_ids2[:, -1] = (token_ids2[:, -1] + 1) % vocab_size     # 只改最后一个 token
logits2 = model(token_ids2)
assert not torch.allclose(logits2, logits)                                   # 最后一位必须变
assert torch.allclose(logits2[:, :-1], logits[:, :-1], atol=1e-5)            # 前面的必须不变
```

同一份权重下追加一个 token，实测前文 logits 的变化量：

```
is_causal=True : 0.0
is_causal=False: 0.155
```

`0.0` 是精确 0 —— 这说明自回归生成「前面写好的内容不会被后面追加的 token 改动」成立；
0.155 意味着每追加一个 token，整段前文的分布都变了，逐步生成无从谈起。

### 5.3 不要靠 loss 判断

做过 150 步对照训练（`is_causal` 唯一变量）：

```
带 mask  : train 6.337 | val 6.775
不带 mask: train 6.344 | val 6.775
```

几乎一样。因为 `d_model=8` 的小模型在 150 步内还没学会「把注意力精确指向 t+1、再让 lm_head
反查词表」这条抄答案的捷径。**loss 曲线分辨不出 mask 有没有接上，必须用 5.1 / 5.2 的结构性断言。**

## 6. 放在哪一层、怎么透传

```
MultiHeadAttention.forward(x, return_attn=False, is_causal=False)
        ↓  self.mha(self.LN1(x), is_causal=is_causal)
TransformerBlock.forward(x, is_causal=False)
        ↓  blk(x, is_causal=is_causal)
MiniGPT.forward(token_ids, is_causal=True)     ← GPT 语义，默认开
```

三个设计决定：

| 决定 | 理由 |
|------|------|
| mask 用 `register_buffer(..., persistent=False)` | 它是常量，能从 `max_len` 推出来。若进 `state_dict`，把 `max_len` 改成 256 再加载旧 checkpoint 就报 `missing key: causal_mask` |
| 预分配 `max_len × max_len`，用时 `[:T, :T]` 切片 | `T ≤ max_len` 恒成立，一次建好；切片是视图，不复制 |
| 超长时按需重建 | 位置编码虽然会先挡住超长序列，但 attention 单独使用时不该假设这一点 |

`is_causal` 只做布尔开关，**不接收 mask 张量**：真要做 padding mask，那个 mask 依赖
每个 batch 的实际长度，属于另一个参数，不该和因果 mask 混在一个入参里。

## 7. 排查清单

```python
# 1. 权重上三角是不是精确 0（是 -inf 顺序问题的唯一直接证据）
i, j = torch.triu_indices(T, T, offset=1)
attn[..., i, j].abs().max()

# 2. 行和是不是 1（先 softmax 后 mask 会掉到 (i+1)/T）
attn.sum(-1)

# 3. mask 有没有进 state_dict（不该进）
[ k for k in m.state_dict() if "causal" in k ]

# 4. 因果性：末尾追加 token，前文 logits 变化量（应精确为 0）
(m(x) - m(torch.cat([x, extra], 1))[:, :x.size(1)]).abs().max()
```

## 心法

1. **mask 是「归一化前排除」，不是「归一化后清零」**：先 `masked_fill(-inf)` 再 `softmax`，顺序反了行和就不是 1。
2. **`triu` 的 `diagonal` 数的是偏移量**：`0` 含主对角线，`1` 不含。留一条对角线是为了不出现全 `-inf` 行（那是 nan）。
3. **`exp(-inf)` 精确等于 0**，所以上三角可以直接断言 `== 0`，不用给容差。
4. **常量 buffer 一定 `persistent=False`**：能从超参推出来的东西不要写进 checkpoint。
5. **因果性要用「改未来、前文不变」来证明**，不要用 loss —— 小模型学不到抄答案的捷径，loss 分不出对错。

> 相关笔记：[training_loop.md](./training_loop.md)（mask 之后怎么把它训练起来）、
> [module_ownership.md](./module_ownership.md)（buffer 与参数的区别）、
> [torch_vs_python.md](./torch_vs_python.md)（`@`、transpose、softmax 轴）。
> 对应代码：[do_attention.py](../do_attention.py)、[transformerBlock.py](../transformerBlock.py)、[main.py](../main.py)。
