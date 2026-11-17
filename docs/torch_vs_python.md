# 原生 Python ↔ PyTorch 写法对照

> 面向「熟悉 Python 原生写法、对 torch 还不熟」的场景，把常用的张量操作和 list 习惯逐一对应起来。
> 结论：日常「读维度」这类小事保持简单即可，真正必须用 torch API 的是矩阵运算/广播这类 list 替代不了的操作。
> 第 7 节起是「矩阵乘 / 维度交换 / 内存布局 / 多头注意力形状骨架」这几类 torch 特有的坑，都是实现 attention 时踩出来的。

## 1. 形状与维度（最容易踩坑）

| 想做的事 | 原生写法 | torch 惯用写法 | 备注 |
|---------|---------|---------------|------|
| 取某维长度 | `len(x)` | `x.size(dim)` / `x.shape[dim]` | 对多维 tensor，`len(x)` **只返回 dim0**，不是总元素数 |
| 取各维长度 | `len(x[0])` | `b, s, d = x.shape` | 解包一次拿全，最直观 |
| 维度个数 | — | `x.dim()` / `len(x.shape)` | |
| 转成 Python list | — | `x.tolist()` | 想彻底用原生逻辑时的安全出口 |

示例：`emb` 形状为 `[2, 3, 8] = [batch, seq_len, d_model]`

```python
emb.size(0)   # 2  -> batch
emb.size(1)   # 3  -> seq_len
emb.size(2)   # 8  -> d_model

len(emb)          # 2  -> 只是 batch 维，语义模糊，不推荐
len(emb[0, :])    # 3  -> 能跑但绕：先切片再取切片后的 dim0
emb.shape[1]      # 3  -> 等价于 emb.size(1)，语义清晰
```

**建议**：看到多维，先想 `.shape` / `.size(dim)` 而不是 `len`。

## 2. 遍历

```python
# 直接 for i in tensor 遍历的是「行」(降一维的子 tensor)，不是标量
for i in x:            # i 是 x 的每一行
    ...

# 想要下标，用 range（对不熟 torch 的人更安全）
for i in range(x.size(0)):
    ...
```

> 历史教训：`for i in pos_encoding[pos]` 里 `i` 是**元素值**（值为 `0.0` 的标量 tensor），
> 又被拿去当写入下标 `pos_encoding[pos, i]`，导致读写指向同一块内存，报错：
> `unsupported operation: ... refer to a single memory location`。
> 正确做法是遍历下标：`for i in range(0, d_model, 2)`。

## 3. 标量数学 vs 张量数学

| 输入类型 | 该用的函数 |
|---------|-----------|
| 纯 `float`/`int` | `math.sin` / `math.cos` |
| `torch.Tensor` | `torch.sin` / `torch.cos` |

```python
torch.sin(0.5)          # ❌ TypeError: must be Tensor, not float
math.sin(0.5)           # ✅
torch.sin(torch.tensor(0.5))  # ✅
```

> 若坚持用 `torch.sin/cos`，把标量先包成 tensor：
> `angle = torch.tensor(pos / (10000 ** (i / d_model)))`。

## 4. Embedding：查表形状

- 表本体 `embedding.weight` 永远是**二维** `[vocab_len, d_model]`。
- 输出形状 = 输入形状，再把最后一维(token id)换成 `d_model`：

| 输入形状 | 输出形状 |
|---------|---------|
| `[]` 标量 | `[d_model]` |
| `[seq]` | `[seq, d_model]` |
| `[batch, seq]` | `[batch, seq, d_model]` |

## 5. 位置编码广播相加

`emb` 是 `[batch, seq, d_model]`，`pos_encoding` 是 `[max_len, d_model]`（按位置查询表，与内容无关）：

```python
pe = pos_encoding[:emb.size(1)]   # 取前 seq 个位置 -> [seq, d_model]
x = emb + pe.unsqueeze(0)          # [b,s,d] + [1,s,d] -> 广播 -> [b,s,d]
```

- `pos_encoding` 只需按 **序列位置** 切片，再靠广播在 batch 维自动复制。
- 约束：`seq_len <= max_len`，否则表不够长得重建更大的表。
- `emb.size(1)` 比 `len(emb[0, :])` 更直接（后者先切片再取 `len`，多一个临时视图）。

### 5.1 广播规则：靠右对齐，不是「自动找同维度」

两个形状**从最右边开始逐维对齐**比较，只有三种结局：

| 情况 | 结果 |
|------|------|
| 两边长度相等 | 通过 |
| 有一边是 `1`，或**某一边根本没有这一维** | 拉伸成对方的长度 |
| 都不满足 | `RuntimeError` |

```
emb  [2, 3, 8]
pe      [3, 8]     ← 右对齐，缺失的 batch 维视作 1
     ─────────
     [2, 3, 8]     ✓  8=8, 3=3, 2 vs (缺失→1) → 拉伸
```

**注意：不是去形状里搜索能对上的维度**，所以形状错时的报错常常很费解 —— 一律按右对齐理解：

```
[2,3,8] + [2,8]  → ERROR: size of a (3) must match size of b (2) at dim 1
[2,3,8] + [8,3]  → ERROR: size of a (8) must match size of b (3) at dim 2
[2,3,8] + [3]    → ERROR: size of a (8) must match size of b (3) at dim 2
```

`[2,8]` 是最容易犯的错：那个 `2` 比的是 `seq_len`，不是 batch。

### 5.2 最危险的是「不报错但算错」

```
[2,3,8] + [1,8]  → [2,3,8]  ✓ 静默通过
```

每个位置加的其实是**同一行**位置编码，程序照跑、照训练，但模型永远学不到位置差异。
`max_len` 一旦退化成 1，`pe[:1]` 就会无声无息地落进这个坑。

所以推荐把要拉伸的维**写出来**，再配一个断言：

```python
seq_len = emb.size(1)
pe = pos_encoding[:seq_len]
assert pe.shape == (seq_len, d_model_len) and seq_len <= max_len
x = emb + pe.unsqueeze(0)      # [1,s,d]，一眼看出 batch 维是拉伸出来的
```

显式 `unsqueeze(0)` 的价值是**可读性 + shape 安全**（对不上立刻报错），不是性能。

### 5.3 `unsqueeze`/`expand` 是零拷贝视图

实测 `pe` 形状 `[3,8]` 时：

```
pe           shape (3, 8)     stride (8, 1)      data_ptr 5232723072
unsqueeze    shape (1, 3, 8)  stride (24, 8, 1)  data_ptr 5232723072   ← 同一块内存
expand(2,…)  shape (2, 3, 8)  stride (0, 8, 1)                        ← batch 维 stride = 0
```

- `unsqueeze` 只改 shape/stride，`data_ptr` 不变，**一个字节都没挪**。
- 广播后 batch 维 stride 是 **0**，含义是「沿这一维走一步跨 0 个元素」，即地址原地不动。
  元素 `[b, t, k]` 的地址 = `t*8 + k`，与 `b` 无关 → 同一份 24 个 float 被读两次、写出 48 个。
- 语义上就是「每个 batch 行加上同一份 `[seq, d_model]`」，但**没有 Python 循环、没有复制**。
- GPU 同理：靠 stride 寻址，`b=1` 的线程命中的正是 `b=0` 刚拉进缓存的同一批数据，因此批量再大也几乎免费。

验证「两行加的是同一份位置编码」：

```python
x = emb + pe.unsqueeze(0)
torch.allclose(x[0] - x[1], emb[0] - emb[1])   # True，pe 被精确抵消
```

### 5.4 `expand` 免费，`repeat` 收费

| 写法 | 结果 |
|------|------|
| `emb + pe` | stride-0 视图，额外内存 0 —— 正确 |
| `emb + pe.unsqueeze(0)` | 同上，额外内存 0，且更易读 |
| `emb + pe.repeat(2, 1, 1)` | **真的分配并拷贝了 48 个 float** |
| `emb + pe.unsqueeze(0).expand(2,-1,-1).clone()` | 同上，白拷贝 |

`repeat()` / `clone()` 会物化副本；`expand()` 和 `+` 里的广播都不会。注意 `+` 本身仍会分配
一个新的输出张量（元素级运算），省掉的只是「广播来源」的那份复制。

## 6. 向量化写法（进阶，替代双重循环）

位置编码的双循环版易读，工程上一般一行搞定：

```python
pos = torch.arange(max_len).unsqueeze(1)   # [max_len, 1]
i = torch.arange(0, d_model, 2)            # [d_model/2]
angle = pos / (10000 ** (i / d_model))     # [max_len, d_model/2]
pe = torch.zeros(max_len, d_model)
pe[:, 0::2] = torch.sin(angle)
pe[:, 1::2] = torch.cos(angle)
```

结果与循环版一致，是熟悉 torch 广播/切片的好练习。

## 7. `@` 矩阵乘：只做乘法，不做转置

一句话规则：**`a @ b` 把 a 的最后一维和 b 的倒数第二维对上求和，其余维度按广播规则右对齐。**

| 操作数形状 | 结果 |
|-----------|------|
| `[k] @ [k]` | 标量 |
| `[m,k] @ [k,n]` | `[m,n]` |
| `[B,H,T,d] @ [B,H,d,T]` | `[B,H,T,T]`（batch 维 `[B,H]` 直接广播，不用循环） |
| `[B,T,d] @ [B,T,d]` | ❌ 要求 `d == T` 才不报错 |

多头注意力打分实测：`Q [1,2,3,4] @ K.transpose(-2,-1) [1,2,4,3] → [1,2,3,3]`，收缩的正是 `d_k=4`。

### 7.1 坑一：把 `@` 当「过一层权重」

`nn.Linear` 内部是 `x @ W.T + b`（`W` 存成 `[out_features, in_features]`）：

```python
import torch.nn.functional as F

self.Wo = nn.Linear(8, 8)
self.Wo(x)                     # ✅ 等价 x @ W.T (+b)
x @ self.Wo.weight             # ❌ 少一个转置，形状对不上
F.linear(x, self.Wo.weight)    # ✅ 与 self.Wo(x) 等价
```

真实踩坑：多头注意力本该 `self.Wo(concat_heads)`，写成了 `concat_heads @ self.Wo(x)`：

- 两边都是 `[B,T,D]`，`@` 要求 `D == T`，`[4,3,8] @ [4,3,8]` 直接报
  `Expected size for first two dimensions of batch2 tensor to be: [4, 8] but got: [4, 3]`
- 若 `seq_len == d_model`（例如 8×8），**不报错**，静默变成「在 token 维上加权求和」——
  这一层的语义彻底变了，而且它照样能训练。

### 7.2 坑二：`Q @ K.T` 而不是 `Q @ K.transpose(-2,-1)`

`.T` 把**所有**维度反转：`[B,H,T,d].T → [d,T,H,B]`，batch/head 全乱。
只转置最后两维必须是 `transpose(-2,-1)`（或 `.mT`）。

## 8. transpose：负编号怎么数

`Q` 形状为 `[B,T,H,d_k]`（实测 `[1,3,2,4]`），目标是「头进 batch」的 `[B,H,T,d_k]`：

| 写法 | 实际交换 | 结果 |
|------|---------|------|
| `x.T` / `x.transpose(-1,-2)` | `d_k ↔ H` | `[B,T,d_k,H]` ❌ |
| `x.transpose(-2,-3)` | `H ↔ T` | `[B,H,T,d_k]` ✅ 但看不出在干什么 |
| `x.transpose(1,2)` | 第 1、2 维 | 同上 ✅ 一眼看懂 |

负编号从右往左数：`-1` 是最后一维，`-2` 是倒数第二……

**建议**：

- 「语义上确实是最后两维」才用负编号——如 `K.transpose(-2,-1)` 转置每个头自己的 `[T,d_k]`，
  与前面有多少个 batch/head 维无关，这时负编号最稳。
- 「把序列维和头维换位置」这类带语义的交换，写正编号 `transpose(1,2)`。
- 同一个表达式里混用 `-2` 和 `-3` 基本等于埋雷。

## 9. stride：`view` 为什么突然失败，`contiguous` 在干什么

张量 = 一块连续内存 + `shape` + `stride`（每一维走一步跨几个元素）。
`transpose` / `view` **只改元数据、不搬数据**。

实测（`B=1, T=3, H=2, d_k=4`，即 `d_model=8`）：

| 张量 | shape | stride | 连续 |
|------|-------|--------|------|
| `torch.randn(1,3,8)` | `(1,3,8)` | `(24,8,1)` | ✓ |
| `.view(1,3,2,4)` 拆头 | `(1,3,2,4)` | `(24,8,4,1)` | ✓ |
| `.transpose(1,2)` | `(1,2,3,4)` | `(24,4,8,1)` | ✗ |
| `weights @ V` 的输出 | `(1,2,3,4)` | `(24,12,4,1)` | ✓ |
| 再 `.transpose(1,2)` | `(1,3,2,4)` | `(24,4,12,1)` | ✗ |
| `.contiguous()` | `(1,3,2,4)` | `(24,8,4,1)` | ✓ |

`view` 是零拷贝的，只有 stride 恰好能「折叠」时才允许。合并相邻两维 `i`、`i+1` 的条件：

```
stride[i] == stride[i+1] * size[i+1]
```

倒数第三行要把 `H(2)` 和 `d_k(4)` 合成 `8`，需要 `12 == 1*4`，不成立：

```
RuntimeError: view size is not compatible with input tensor's size and stride (...).
Use .reshape(...) instead.
```

**为什么 matmul 之后反而更糟**：`weights @ V` 产生的是一个新张量，按 `[B,H,T,d_k]` 顺序连续。
所以 transpose 回来后，内存里是「先排完 head0 的所有 token，再排 head1」，
同一 token 的两个头隔了 12 个 float；而 `[B,T,8]` 要求同一 token 的 8 个数紧挨着。

### 9.1 什么时候「transpose 后 view 又能用」

只有 transpose 恰好是在**撤销**一次 view 时才成立（因为 stride 又兼容了）：

```python
y = x.view(1,3,2,4).transpose(1,2).transpose(1,2)   # stride 回到 (24,8,4,1)
y.view(1,3,8)      # ✅ 等价于撤销那次拆头
```

中间一旦经过 matmul 之类产生新张量，这条路就断了。

### 9.2 三种写法怎么选

| 写法 | 拷贝 | 什么时候用 |
|------|------|-----------|
| `.view(...)` | 不拷贝 | 明确知道连续（刚从 `nn.Linear` 出来、刚 `contiguous` 过） |
| `.reshape(...)` | 需要时拷贝 | **默认选它**：能 view 就 view，不行就自动 copy |
| `.contiguous().view(...)` | 强制拷贝 | 想让「这里确实在重排内存」在代码里显式可见 |

补充：

- 张量本来就连续时 `.contiguous()` 直接 `return self`，不拷贝，写一句不花性能。
- 这次拷贝只是内存带宽开销，数值、dtype、梯度都不变。
- 别靠猜报错，直接打：`print(x.shape, x.stride(), x.is_contiguous())`。

## 10. 多头注意力的完整形状骨架

以 `d_model=8, num_heads=2 → d_k=4` 为例：

| 步骤 | 代码 | shape |
|------|------|-------|
| 输入 | `x` | `[B,T,8]` |
| QKV 投影 | `self.Wq(x)` | `[B,T,8]` |
| 拆头 | `.view(B,T,H,d_k)` | `[B,T,2,4]` |
| 头进 batch | `.transpose(1,2)` | `[B,2,T,4]` |
| 打分 | `Q @ K.transpose(-2,-1)` | `[B,2,T,T]` |
| 归一化 | `F.softmax(..., dim=-1)` | `[B,2,T,T]`（每个 query 对所有 key 和为 1） |
| 加权 | `@ V` | `[B,2,T,4]` |
| 回到 seq 序 | `.transpose(1,2)` | `[B,T,2,4]` |
| 合并头 | `.contiguous().view(B,T,8)` | `[B,T,8]` |
| 输出投影 | `self.Wo(...)` | `[B,T,8]` |

这个骨架里实际踩过的四个坑：

### 10.1 `d_k` 用了真除法 → 静默变 float

```python
self.d_k = d_model_len / num_heads    # 4.0，float
Q.view(B,T,H,self.d_k)                # TypeError: ... type must be tuple of ints, but got float
```

用 `//`，并且要在**构造时**断言整除（否则 `8 // 3 == 2`，要到 `view` 才炸，报错指不到原因）：

```python
assert d_model_len % num_heads == 0, f"{d_model_len} 不能被 {num_heads} 整除"
self.d_k = d_model_len // num_heads
```

### 10.2 变量名遮蔽：`B, T, C = x.shape` 里的 `C` 是 d_model

```python
B, T, C = x.shape          # C = 8，是 d_model
Q.view(B, C, H, d_k)       # 把 d_model 当成了序列长度
# RuntimeError: shape '[4, 8, 2, 4]' is invalid for input of size 96
```

`C` 是 PyTorch 惯例的 channel/feature 维，放到「序列」的位置上就是 bug。
建议 `B, T, D = x.shape`，让 `view(B,T,H,d_k)` 读起来无歧义。

### 10.3 `x.shape` 解包前先断言维数

```python
batch, seq_len, d = x.shape
# 传 2D 时：ValueError: not enough values to unpack (expected 3, got 2)
```

这个报错完全不提「第几维不对」。先断一句：

```python
assert x.dim() == 3, f"要求 [batch, seq_len, d_model]，实际 {tuple(x.shape)}"
```

### 10.4 `softmax` 的 `dim` 与缩放的 `sqrt(d_k)`

- `F.softmax` 必须显式 `dim=-1`。多头把 shape 变成 `[B,H,T,T]` 后，
  「最后一个维度」才是 key 维；省略参数在多了 batch/head 维时会沿错轴归一化。
- 缩放因子是 `d_k`（每个头只累加 `d_k` 项），不是 `d_model`。
  写成 `sqrt(d_model)` 会让 logits 偏小、梯度偏弱 —— 不报错的数值错误。

## 11. 调用模块：`M(x)` 而不是 `M.forward(x)`

```python
M(x)            # ✅ 走 __call__ → forward + 各种 hook
M.forward(x)    # ⚠️ 能跑，但绕过 hook（profiler、hook 埋点、部分封装都会失效）
```

顺手在 `__main__` 里放一句形状断言，是这类问题唯一低成本的防线：

```python
out = M(torch.randn(4, 3, 8))
assert out.shape == (4, 3, 8), f"应输出 (4,3,8)，实际 {tuple(out.shape)}"
```

## 12. 布尔掩码：`triu` + `masked_fill` + `-inf`

因果 mask、padding mask 都是这一套，三个元素：**bool mask、`masked_fill`、`-inf`**。

### 12.1 造掩码：`triu` 只认最后两维

```python
w = torch.arange(1., 17.).view(1, 1, 4, 4)      # 4D 也能用
torch.triu(w, diagonal=1)[0, 0].int().tolist()
# [[0, 2, 3, 4], [0, 0, 7, 8], [0, 0, 0, 12], [0, 0, 0, 0]]
```

和 `softmax(dim=-1)` 一样，`triu` / `tril` 是**只认末尾两维**的家族：前面有几维
batch/head 都不影响。所以 `[B,H,T,T]` 的注意力权重可以直接 `torch.equal(attn, attn.tril())`
判断是不是下三角，不需要循环。

`dtype` 要显式给，否则拿到的是 float 不是 mask：

```python
torch.ones(4, 4)                     # float32，全是 1.0
torch.ones(4, 4, dtype=torch.bool)   # bool，全 True —— 这才是 mask
```

`diagonal` 是偏移量：`0` 含主对角线，`1` 不含。做因果 mask 要用 `1`，
否则第 0 行整行 `-inf`，softmax 得到 nan。

### 12.2 检查掩码：`triu_indices` + 高级索引

```python
i, j = torch.triu_indices(4, 4, offset=1)
# i = [0, 0, 0, 1, 1, 2]
# j = [1, 2, 3, 2, 3, 3]

a = torch.arange(16.).view(1, 4, 4)
a[..., i, j]     # tensor([[1., 2., 3., 6., 7., 11.]])   shape (1, 6)
```

两个 index 张量做高级索引，`...` 保留前面所有维，结果形状 = `前面的维 + index 的形状`。
断言“上三角全为 0”时比 `(a * mask).abs().max()` 更方便：不用先把 bool 转成 float。

### 12.3 `masked_fill` 的 mask 必须是 bool

```python
torch.ones(2, 2).masked_fill(torch.ones(2, 2), 0.0)
# RuntimeError: masked_fill_ only supports boolean masks, but got mask with dtype float
```

语义：**mask 为 True 的位置填 value**，False 保持原值。
`masked_fill` 返回新张量；`masked_fill_` 原地改（会把版本号 +1，见 §15.5）。

### 12.4 `-inf` 与 `exp(-inf)`

```python
s = torch.tensor([[1., 2., 3., 4.]])
mask = torch.triu(torch.ones(1, 4, dtype=torch.bool), 1)
s.masked_fill(mask, float("-inf"))        # [[1.0, -inf, -inf, -inf]]
torch.softmax(s.masked_fill(mask, float("-inf")), -1)   # [[1.0, 0.0, 0.0, 0.0]]
```

- `exp(-inf) == 0` 是**精确**的 0 → 断言直接写 `== 0`，不用给容差。
- 整行都是 `-inf` → `nan`（不是 0）：`softmax([-inf, -inf, -inf]) = [nan, nan, nan]`。
- 顺序：先 fill 再 softmax。先 softmax 再置零，每行权重和就不再是 1。详见 [causal_mask.md](./causal_mask.md)。

## 13. dtype、相等性断言、浮点分辨率

### 13.1 索引张量必须是整数

```python
nn.Embedding(10, 4)(torch.tensor([[1.0, 2.0]]))
# RuntimeError: Expected tensor for argument #1 'indices' to have one of the following
# scalar types: Long, Int; but got torch.FloatTensor instead ...
```

`torch.tensor([encode(t)])` 得到的是 `int64`，正好；从 numpy / float 转过来的要 `.long()`。
模块里可以先断一句：`assert not torch.is_floating_point(ids)`。

### 13.2 `torch.equal` 判“完全一样”，`torch.allclose` 判“数值接近”

| | 判据 | 典型用途 |
|---|---|---|
| `torch.equal(a, b)` | 逐元素全等，且 shape 全等 | 可复现性（同输入两次前向） |
| `torch.allclose(a, b)` | `\|a-b\| <= atol + rtol*\|b\|` | 与参考实现 / 手写前向对齐 |

```python
torch.equal(model(ids), model(ids))        # True/False，没有容差
torch.allclose(out, manual, atol=1e-6)     # 有浮点误差时的正确选择
```

“同输入两次结果不同”这类 bug **只有 `equal` 抓得住** —— `allclose` 会给它容差混过去。
对 bool 张量 `allclose` 也能算（同值 True、异值 False），但没有“接近”的语义，用 `equal` 更明确。

### 13.3 float32 只有 7 位有效数字

```python
torch.equal(torch.tensor([0.1]), torch.tensor([0.1 + 1e-9]))   # True —— 是同一个数
torch.finfo(torch.float32).eps                                  # 1.1920928955078125e-07
```

推论：

- 想“制造一点点差异”（比如测 LayerNorm 的 eps），`1e-9` 不够，至少要 `1e-7` 量级；
- 方差 / 差分在近似常数上会悄悄变成 0 或 `1e-15`；
- 数值断言一律给 `atol`，只有“应当精确为 0”的场合例外（`exp(-inf)`、mask 置零）。

## 14. 注册与账本：谁在 `parameters()` 里

### 14.1 容器决定注册

| 写法 | `parameters()` | `state_dict()` | `.to(device)` / `.eval()` |
|------|---------------|---------------|--------------------------|
| `[Block(), ...]`（普通 list） | ✗ | ✗ | ✗ |
| `nn.ModuleList([...])` | ✓ | ✓ | ✓ |
| `nn.Sequential(...)` | ✓ | ✓ | ✓ |

普通 list 里的模块 Python 层面能跑，PyTorch 完全看不见它们。

### 14.2 `buffer` 与 `persistent`

```python
self.register_buffer("pe", table)                            # 进 state_dict，但不进优化器
self.register_buffer("causal_mask", mask, persistent=False)   # 不进 state_dict
```

两种都跟着 `.to(device)` 搬家；区别只在 checkpoint 里有没有它。
能从超参推出来的常量（mask）用 `persistent=False`，否则改 `max_len` 后加载旧权重会报缺 key。

### 14.3 四个查询接口

| 调用 | 给什么 |
|------|-------|
| `model.parameters()` | 只要张量 |
| `model.named_parameters()` | 名字 + 张量 |
| `model.state_dict()` | 参数 **+ buffer** |
| `model.named_modules()` | 结构树（查重复注册用） |

实测 MiniGPT（`d_model=8, num_blocks=12`）：

```
state_dict 键数: 149 | parameters 数: 148
差集（只在 state_dict 里）: ['pos_encoding.pe']
```

### 14.4 重复注册：`Sequential` + 同名属性

```python
s.a = nn.Linear(2, 2)
s.s = nn.Sequential(s.a)
# state_dict: ['a.weight', 'a.bias', 's.0.weight', 's.0.bias']   4 个键，2 份权重
# parameters(): 2 个（去重）
```

checkpoint 体积翻倍，`named_modules()` 里同一层出现两次。详见 [module_ownership.md](./module_ownership.md)。

## 15. 训练循环里最常用的几个调用

### 15.1 `var` 默认 `unbiased=True`

```python
x = torch.arange(8.).view(1, 8)
x.var(-1)                    # 6.0     分母 n-1 = 7（无偏）
x.var(-1, unbiased=False)    # 5.25    分母 n = 8（总体）
# 6.0 / 5.25 = 1.1428 = 8/7
```

LayerNorm 要的是总体方差，必须显式 `unbiased=False`；否则归一化输出整体缩小约 6.5%（n=8）。
`x.std(...)` 同理（它只是 `var` 开根号）。

### 15.2 `F.cross_entropy` 只认 `[N, C]` 和 `[N]`

```python
F.cross_entropy(torch.randn(2, 3, 10), torch.randint(0, 10, (2, 3)))
# RuntimeError: Expected target size [2, 10], got [2, 3]
```

报错里的 `[2, 10]` 说明它把 `dim=1`（=3）当成了类别维。修法是先压平：

```python
F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=pad_token)
```

`ignore_index` 让指定 id 的位置不参与 loss（处理 padding 的标准做法）。

### 15.3 梯度三件套

```python
optimizer.zero_grad(set_to_none=True)      # grad 变 None（不是变 0）
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 返回裁剪前的总范数（实测 2.4495）
[n for n, p in model.named_parameters() if p.grad is None] # 漏注册 / 断链
```

`set_to_none=True` 之后 `p.grad is None` 既表示“没反传过”、也表示“刚被清空”，
所以“检查梯度覆盖”的断言必须放在 `backward()` **之后**。

### 15.4 `eval()` 和 `no_grad()` 是两件事

| 只做一半 | 后果 |
|---------|------|
| 只 `eval()` | 仍在建计算图：内存翻倍、变慢 |
| 只 `no_grad()` | dropout / BN 仍在训练模式，评估结果抖动 |
| 两个都做 | 评估的正确姿势；结束后 `model.train()` 切回 |

### 15.5 原地改参数 + 对旧输出反传 = version error

```python
out = (x @ W).sum()          # 计算图里记下了 W
with torch.no_grad():
    W.zero_(); W.add_(1.0)   # 版本号 0 → 2
out.backward()
# RuntimeError: one of the variables needed for gradient computation has been modified
# by an inplace operation: [torch.FloatTensor [2, 2]] is at version 2; expected version 0 instead.
```

触发条件：被改的张量**在反传路径上还要被用到**（比如要算对输入的梯度）。
只在 `nn.Linear` 上改权重、而输入不需要梯度时，autograd 会跳过这项检查、**不报错** ——
所以这个错误会“有时出现有时不出现”，最难排查。

修法：改完（或恢复完）**重新跑一次前向**，用新的计算图反传。

## 16. 采样：`topk` + `softmax` + `multinomial`

生成一个 token 的最小组合：

```python
logits = model(ids)[:, -1, :] / temperature          # [B, V]
kth = torch.topk(logits, top_k).values[:, -1:]       # [B, 1]，第 k 大的值
logits = logits.masked_fill(logits < kth, float("-inf"))   # bool mask，复用 §12.3
probs = torch.softmax(logits, dim=-1)                # [B, V]
nxt = torch.multinomial(probs, num_samples=1)        # [B, 1]，dtype=int64
ids = torch.cat([ids, nxt], dim=1)                   # [B, T+1]
```

实测行为：

| 调用 | 结果 |
|------|------|
| `torch.topk(x, 3).values[:, -1:]` | `[B,1]`，第 k 大的值（走 top-k 阈值） |
| `logits < kth` | `[B,V]` 的 bool mask（`[B,V]` 与 `[B,1]` 广播） |
| `torch.multinomial(p, 1)` | `int64`，可直接 `cat` 到 ids 后面 |
| 权重不归一（和为 6） | 也接受，按权重比例抽 |
| 权重含负数 | `RuntimeError: probability tensor contains either \`inf\`, \`nan\` or element < 0` |

`cat` 与 `stack`：

```python
torch.cat([a, b], 1)     # 沿已有维度拼接   [2,1] + [2,1] → [2,2]
torch.stack([a, b], 1)   # 新建一维         [2,1] + [2,1] → [2,2,1]
```

生成时的滑窗 `ids[:, -max_len:]` 也是切片视图，不是算术：位置编码表只有 `max_len` 行。

---

## 核心心法

1. **多维优先用 `.shape` / `.size(dim)`，别用 `len`**（多维 tensor 上 `len` 只给 dim0）。
2. **遍历下标用 `range(x.size(0))`**，避免 `for i in tensor` 拿到值/行造成误用。
3. **标量数学用 `math.*`，张量数学用 `torch.*`**，两者不能混用。
4. 保持原生逻辑没问题，只在「读维度、矩阵运算、广播」这类地方用 torch，迁移成本最低。
5. **广播靠右对齐，靠猜一定会错**：想要拉伸的那一维用 `unsqueeze(0)` 显式写出来，shape 不对立刻报错，而不是静默算错。
6. **`unsqueeze` / `expand` 零拷贝（stride 0），`repeat` / `clone` 才会真复制**：广播本身不花内存，花内存的只有最终的输出张量。
7. **`@` 只是矩阵乘**：它不会帮你转置，也不是「过一层权重」。`nn.Linear` = `x @ W.T + b`（`W` 存成 `[out,in]`）。
8. **transpose/permute 之后一律用 `reshape`**：`view` 要求 stride 兼容，不兼容就报错；怀疑布局就打印 `stride()` / `is_contiguous()`。
9. **换维用正编号写明**（`transpose(1,2)`），负编号只在「语义确实是最后两维」时用（`K.transpose(-2,-1)`）。
10. **形状错误先怀疑三件事**：真除法该用 `//`、变量名遮蔽（`C` 还是 `T`）、`x.shape` 解包顺序 —— 先断言再算，断言带消息。
11. **掩码三件套：bool mask + `masked_fill` + `-inf`**：`triu/tril` 只认末尾两维，`diagonal` 数的是偏移量，`exp(-inf)` 是精确 0、整行 `-inf` 是 nan。
12. **float32 只有 7 位有效数字**：`0.1 + 1e-9 == 0.1`，数值断言一律给 `atol`，只有“应当精确为 0”才写 `== 0`。
13. **`equal` 判可复现，`allclose` 判数值对齐**：前者没有容差，后者会把“结果不一致”当误差放过。
14. **`parameters()` 管训练、`state_dict()` 管保存、`ModuleList` 才注册**：buffer 用 `register_buffer`，能推断的常量加 `persistent=False`。
15. **训练四件套**：`reshape(-1, V)` 喂 `cross_entropy`、`var` 用 `unbiased=False`、`clip_grad_norm_` 有返回值、`eval()` 与 `no_grad()` 各管一摊。
16. **原地改过参数就别拿旧输出反传**：`version error` 只在反传路径还需要那个张量时才报，改完重新前向一次。

> 相关笔记：同一个单词（如 `cat`）在不同位置拿到不同 embedding 的问题，见 [tokenizer_whitespace.md](./tokenizer_whitespace.md)。
> 第 7-16 节对应的代码：[../do_attention.py](../do_attention.py)、[../train.py](../train.py)、[../main.py](../main.py)。
>
> 模型结构层面的坑（形状都对、数值/归属错）：
> - 因果 mask 的顺序与 `diagonal`：[causal_mask.md](./causal_mask.md)
> - 参数 / buffer / ModuleList 的归属：[module_ownership.md](./module_ownership.md)
> - 残差基准与 LayerNorm 的 eps：[residual_and_layernorm.md](./residual_and_layernorm.md)
> - 把模型真正训起来：[training_loop.md](./training_loop.md)
> - 全部文档的索引：[README.md](./README.md)
