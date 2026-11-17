# 残差与 LayerNorm：两个"形状对、数值错"的经典坑

> 起因：`transformerBlock.py` 的第二次残差写成了 `x + ffn(...)`，`do_LayerNorm.py` 写成了 `x - mean/std`。
> 两处都**不报错**，形状全对，loss 也能降。
> 结论：残差的基准必须随子层推进（`h1 → out`），LayerNorm 的分母必须是 `sqrt(var + eps)`。
> 这两个错误的共同点是 —— 只有数值层面的对照实验才能抓出来。

## 1. 残差：两次残差的基准不一样

```python
h1 = x + self.mha(self.LN1(x))          # 基准是 x
return h1 + self.ffn(self.LN2(h1))      # 基准是 h1，不是 x
```

```
x ──┬──► LN1 ─► MHA ─┐
    │                ├─(+)─► h1 ──┬──► LN2 ─► FFN ─┐
    └────────────────┘            │                 ├─(+)─► out
                                  └─────────────────┘
```

写错的那一版：

```python
return x + self.ffn(self.LN2(h1))       # ❌ 注意力只通过 h1 → LN2 → FFN 这条间接路径影响输出
```

### 1.1 实验一：把 FFN 输出置零

`linear2` 的 weight 和 **bias** 都要清零（只清 weight 会留下偏置，FFN 输出不为 0）：

```python
blk.ffn.linear2.weight.zero_()
blk.ffn.linear2.bias.zero_()
```

实测：

```
FFN 输出置零后：正确版 == h1 ? True
FFN 输出置零后：错误版 == x  ? True
```

`True` 是 `torch.allclose(..., atol=1e-6)` 的精确通过 —— 这说明错误版里 `h1` 完全没进输出，
注意力子层等于白算。

### 1.2 实验二：把注意力输出换成 0

把 `self.mha` 换成一个恒返回 `zeros_like(x)` 的模块，量"最终输出变化了多少"：

```
||attn|| = 1.7997
注意力被置零后，输出变化量 / ||attn||：
  正确版 (h1 基准): 1.0396
  错误版 (x 基准) : 0.2203
```

- 正确版 ≈ 1：注意力的贡献**系数是 1**，原样进主干（多出来的 0.04 是 FFN 跟着 `LN2(h1)` 一起变的部分）。
- 错误版 ≈ 0.22：注意力的贡献被压到自身的 22%。差值来自 `FFN(LN2(h1)) - FFN(LN2(x))`，
  也就是**系数变成了 FFN 的雅可比 `J_ffn`**，而不是 1。

这不是"没有贡献"，是"贡献被一个随机权重矩阵打了折"。而且 `LN2` 会重新归一化幅度、`ReLU` 会截断负值，
所以这个折扣既不恒定也不可解释 —— 剩下的只是"还能训"的假象。

### 1.3 残差必须写在 block 层，不能写进子层

在 `MultiHeadAttention.forward` 里加 `attention_out = attention_out + x`，然后 block 里再加一次残差，
等于同一个 `x` 被加了两次：

- 残差主干上出现 **2×x** 的偏置，随着 12 层堆叠不断放大；
- 单头 `Attention` 和多头 `MultiHeadAttention` 的语义就不一致了（一个自带残差、一个没有），
  两者无法替换；
- 真正该由 block 统一管理的结构（Pre-LN / Post-LN、残差缩放、dropout）被拆散到两个文件里。

**约定：子层的输出永远是"增量"，残差只加一次，且由父模块加。**

## 2. LayerNorm：三个坑，一个比一个隐蔽

```python
mean = x.mean(-1, keepdim=True)
var = x.var(-1, keepdim=True, unbiased=False)     # ① 总体方差
x_norm = (x - mean) / torch.sqrt(var + self.eps)  # ② eps 在根号里
return self.gamma * x_norm + self.beta
```

### 2.1 运算符优先级：`x - mean / std`

```python
x - mean / std      # ❌ 先算 mean/std，再整体减掉
(x - mean) / std    # ✅
```

两个版本形状完全相同，只是数值不同：

```
x - mean/std 的最大值: 13.97 | 与 nn.LayerNorm 最大误差: 12.03
输入 std: 4.4195        | 错误版本输出 std: 4.4195      ← 一模一样：根本没归一化
```

后面那个数字才是判据：**归一化之后输出 std 必须 ≈ 1**。写错时输出 std 就等于输入 std，
逐元素只是减了一个常数 —— 模型仍然能训，但深度带来的收益基本没了。

### 2.2 `unbiased=True`（PyTorch 默认）

`x.var(-1)` 默认 `unbiased=True`，分母是 `n-1`；LayerNorm 要的是总体方差（分母 `n`）：

```
n=8 时 std 被放大 sqrt(8/7) = 1.0690
→ 归一化输出整体缩小约 6.5%
```

实测与 `nn.LayerNorm` 的最大误差：`0.1327`。而加上 `unbiased=False` 之后：`2.38e-07`。
**6 个数量级的差距**，而且是纯系统偏差（所有位置一起偏小），比随机误差更伤训练。

### 2.3 `eps` 缺失 → nan 与梯度爆炸

```python
torch.sqrt(var)          # ❌ 方差为 0 时除 0
torch.sqrt(var + self.eps)   # ✅
```

两层保护：

**（a）常数行 → nan。** 同一行 8 个数完全相等时 `var = 0`，`0/0`：

```
eps=0    输出=[[nan, nan]]   是 nan: True
eps=1e-6 输出=[[0.0, 0.0]]   是 nan: False
```

真实训练里 padding 位置、或者 float32 分辨率以下的差异都会出现这种行。
顺带一个事实：`0.1` 和 `0.1 + 1e-9` 在 float32 里**是同一个数**（`float32 eps ≈ 1.19e-7`），
所以"我给了微小差异"并不能保证 `var > 0`。

**（b）近似常数行 → 梯度爆炸。** 方差 2.36e-15 时（不是 0，不会 nan）：

```
eps=0    梯度最大 = 1.49e+06
eps=1e-6 梯度最大 = 8.50e+02        ← 小了 1750 倍
```

`d/dx` 里有 `1/sqrt(var+eps)` 这一项，`var → 0` 时它趋于无穷。eps 的作用就是给这个分母设下限，
把梯度锁在可训练的区间里。这也是为什么 GPT 系列的 eps 通常取 `1e-5 ~ 1e-6` 而不是 `1e-8`。

### 2.4 `sqrt(var + eps)` 而不是 `sqrt(var) + eps`

以 `var = 1e-8`、`eps = 1e-6` 为例：

| 写法 | 分母 |
|------|------|
| `sqrt(var + eps)` | `0.001005` |
| `sqrt(var) + eps` | `0.000101` |

两者差 10 倍。当 `var ≫ eps` 时两种写法都退化到 `sqrt(var)`，看不出区别；
只有 `var` 和 `eps` 同量级时才暴露 —— 而那种情况恰恰就是需要 eps 的时候。
理论公式里 eps 写在方差上（`Var[x] + ε`），代码就该跟着写在根号里面。

### 2.5 有 eps 就一定对不齐 `nn.LayerNorm`？

对齐了也要注意残差量级：`var/(var+eps) < 1`，输出 std 会比 1 略小。
本仓库 `d_model=8` 的实测偏差是 `var(-1)` 与 1 相差 `1.37e-06` 量级，属于正常。

## 3. 对照 `nn.LayerNorm` 的自测

```python
ln = LayerNorm(d_model_len)
ref = nn.LayerNorm(d_model_len, eps=ln.eps)
assert torch.allclose(ln(x), ref(x), atol=1e-5)
assert ln(x).mean(-1).abs().max() < 1e-6          # 每行均值 ≈ 0
assert (ln(x).var(-1, unbiased=False) - 1).abs().max() < 1e-4   # 每行方差 ≈ 1
assert torch.isfinite(ln(torch.full((2, 3, 8), 0.5))).all()     # 常数行不 nan
```

配套的残差自测（`transformerBlock.py` 的 `__main__`）：

```python
h1 = x + blk.mha(blk.LN1(x))
manual = h1 + blk.ffn(blk.LN2(h1))
assert torch.allclose(blk(x), manual, atol=1e-6)     # 手写复现，抓"基准写错"
assert not torch.allclose(blk(x), x)                 # 子层真的起作用了
assert blk.LN1(x).mean(-1).abs().max() < 1e-6        # Pre-LN：进子层前均值≈0
```

## 4. 排查清单

```python
# 残差
h1 = x + blk.mha(blk.LN1(x))
torch.allclose(blk(x) - h1, blk.ffn(blk.LN2(h1)))        # True = 第二次残差基准正确

# LayerNorm：输出统计量比"看着像"靠谱
out = ln(x)
out.mean(-1).abs().max(), (out.var(-1, unbiased=False) - 1).abs().max()

# 常数行
torch.isnan(ln(torch.full((2, 8), 3.0))).any()

# 与官方实现对照
nn.LayerNorm(8, eps=1e-6)(x)
```

## 心法

1. **残差基准逐级推进**：`h1 = x + f(x)`，`out = h1 + g(h1)`。第二次写回 `x` 不报错，但注意力贡献系数会从 1 变成 `J_ffn`。
2. **子层只输出增量，残差由父模块加一次**：否则残差主干上同一个 `x` 被重复累加，越堆越偏。
3. **判断残差对不对，靠"置零 + 对照手写"，不靠 loss**：把 FFN 置零看输出是不是 `h1`，是最便宜的证据。
4. **LayerNorm 先看输出 std 是不是 ≈1**：`x - mean/std` 的优先级错误会让输出 std 完全等于输入 std。
5. **`var` 用 `unbiased=False`**：默认值会让输出整体缩小 6.5%（n=8），差值 6 个数量级。
6. **`eps` 写在根号里，取 1e-5~1e-6**：管 nan，也管梯度爆炸（实测 1.49e6 → 8.50e2）。

> 相关笔记：[module_ownership.md](./module_ownership.md)、[torch_vs_python.md](./torch_vs_python.md)（形状/广播类基础）。
> 对应代码：[transformerBlock.py](../transformm/transformerBlock.py)、[do_LayerNorm.py](../transformm/do_LayerNorm.py)、[do_ffn.py](../transformm/do_ffn.py)。
