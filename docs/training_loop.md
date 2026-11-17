# 训练循环：一条样本是「错位一位」的序列

> 起因：给 MiniGPT 加了 [train.py](../transformm/train.py)，300 步把 loss 从 11.5 降到 6.66。
> 结论：自回归训练只有两个必须做对的动作 —— **标签右移一位**、**注意力开因果 mask**。
> 其余（lr、clip、batch）都是调参；这两个错了则不报错、loss 照降、模型学废。

## 1. 一条样本长什么样

```python
x = ids[i     : i + T]      # 输入，T=32
y = ids[i + 1 : i + T + 1]  # 标签：y[t] 就是 x[t] 位置的"答案"
```

```
ids:  ... the  key  to  building  rapport  is  ,  to  ...
x   :      the  key  to  building  rapport  is
y   :           key  to  building  rapport  is  ,
```

所以**同一个位置 t 既在输入里、又在标签里**（标签是它的下一个 token），
`seq_len=32` 的一条样本贡献 **32 个**训练信号，不是 1 个。

形状链条：

| 步骤 | shape |
|------|-------|
| `x` / `y` | `[B, T]`，`B=8, T=32` |
| `model(x)` → logits | `[B, T, V]`，`V=100277` |
| 压平 | `[B*T, V]` 与 `[B*T]` |
| `cross_entropy` | 标量 |

`get_batch` 里起点上限要留够位置，否则 `y` 切片越界：

```python
starts = torch.randint(0, data.numel() - block_size - 1, (batch_size,))
```

## 2. 交叉熵必须压平，维度对不上不报错

```python
def lm_loss(logits, y):
    assert logits.shape[:2] == y.shape, ...
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),   # [B,T,V] → [B*T, V]
        y.reshape(-1),                    # [B,T]   → [B*T]
        ignore_index=IGNORE_INDEX,        # pad 位置不计入 loss
    )
```

三个容易错的地方：

- `logits` 直接传 `[B,T,V]` 会报 `Expected input batch_size (…) to match target batch_size (…)` ——
  但报错里的数字是 `B` 和 `B*T`，看着像 batch 没对上，实际是**忘了压平**。
- `y` 忘了 `reshape(-1)` 且 `T==V` 时可能静默广播错。
- `ignore_index=pad_token` 只有在标签里真的出现过 pad 才有意义；本仓库语料是整段拼接、无 pad，
  写着是为了后面加 padding 时不用改函数。

## 3. 用 `ln(V)` 当基线，比"loss 看着降了"靠谱

随机初始化时，每个位置在 100277 个候选里基本均匀乱猜，交叉熵期望就是：

```
ln(100277) = 11.516
```

实测起点正是这个量级：

```
参数量 1614528 | 随机初始化期望 loss ≈ ln(100277) = 11.516
step    1 | step loss  11.738 | train  11.634 | val  11.627 |   1.0s
```

train.py 结尾就断言 `train_loss < ln(V) - 0.5`。这条断言能挡住一整类 bug：

| 现象 | 大概率是 |
|------|---------|
| loss 停在 11.5 不动 | lr 太小 / 梯度断了 / mask 把一切都挡住了 |
| 初始 loss ≈ 4.6（= `ln(100)`） | 词表维度接错（比如 logits 最后一维是 d_model） |
| 初始 loss ≈ 0.69（= `ln(2)`） | 标签被当成二分类，或 logits 被 softmax 了两次 |
| loss 是 nan | 学习率过大 / 没做梯度裁剪 / LayerNorm eps 缺失 |

## 4. 超参与实测曲线

```bash
python -m transformm.train                        # 默认 300 步
python -m transformm.train --steps 100 --block-size 64
```

| 超参 | 默认 | 为什么 |
|------|------|--------|
| `batch_size` | 8 | logits 是 `[8,32,100277]` float32 ≈ **103 MB**，再大 CPU 上吃内存 |
| `block_size` | 32 | 上限是 `max_len=109`；越长每步越慢，但梯度噪声更小 |
| `lr` | 3e-3 | 试过 1e-3：20 步只到 11.45，300 步来不及。d_model=8 需要更大的步长 |
| `betas` | (0.9, 0.95) | GPT 系列惯例，二阶动量衰减更快，适合稀疏梯度（embedding） |
| `weight_decay` | 0.01 | AdamW 的解耦权重衰减，抗过拟合 |
| `clip_grad_norm_` | 1.0 | 小模型 + 高 lr 偶发梯度爆炸，裁到范数 1.0 |
| `eval_batches` | 10 | 评估也只是抽样，10 批足够看出趋势 |

实测（CPU，单进程约 0.14 s/step；同一 seed 跑两次数字完全一致，见 [checkpoint.md](./checkpoint.md)）：

```
step    1 | step loss  11.738 | train  11.634 | val  11.607 |   1.1s
step   50 | step loss   9.761 | train   9.727 | val   9.855 |   7.1s
step  100 | step loss   7.136 | train   7.171 | val   7.580 |  13.3s
step  150 | step loss   6.279 | train   6.337 | val   6.942 |  19.5s
step  200 | step loss   6.174 | train   6.216 | val   6.701 |  26.8s
step  250 | step loss   5.949 | train   6.096 | val   6.618 |  33.8s
step  300 | step loss   5.945 | train   6.033 | val   6.521 |  42.2s
最终 train 6.033 | val 6.521 | 随机猜测 11.516 | 提升 4.995 nats
```

读法：

- 前 100 步降得最快（学会高频 token 的一元统计，比如标点、` the`、` to`）；
- 100 步之后 train 稳定下降，val 每次只降 0.08–0.10 —— **收益递减，再练下去就是过拟合**；
- 300 步时 train 与 val 差 0.49 nat，属于正常（训练集 7 万 token、模型 1.6M 参数）。

## 5. 评估和训练的开关不能混

```python
@torch.no_grad()
def estimate_loss(...):
    rng_state = torch.get_rng_state()     # 评估抽样会消费 RNG，先存一份
    model.eval()          # 关掉 dropout，否则每次评估的 loss 都在抖
    ...
    model.train()         # 一定要切回来，否则后续训练不再 dropout
    torch.set_rng_state(rng_state)        # 恢复 RNG：评估对训练“无副作用”
```

`model.eval()` 和 `torch.no_grad()` 是**两件不同的事**，缺一个都有坑：

| 只用一样 | 后果 |
|---------|------|
| 只 `eval()` | 仍会建计算图，评估内存翻倍、变慢 |
| 只 `no_grad()` | dropout 还在生效，评估 loss 抖动、与训练不一致 |

## 6. 生成：每一步只取最后一个位置

```python
for _ in range(new_tokens):
    window = ids[:, -max_len:]                    # 位置编码只有 max_len 个位置
    logits = model(window)[:, -1, :] / temperature
    nxt = torch.multinomial(torch.softmax(logits, -1), 1)
    ids = torch.cat([ids, nxt], dim=1)
```

四个要点：

1. **只取 `[:, -1, :]`**：前面位置的 logits 我们不关心，但在因果 mask 下它们也不会因新 token 而变。
2. **滑窗 `ids[:, -max_len:]`**：位置编码表只有 `max_len=109` 行，超了会触发断言。
3. **`top_k` 过滤**：把小于第 k 大概率的位置置 `-inf`，避免从 10 万个几乎等概率的候选里抽到乱码。
4. **没有 KV cache**：每步重算整段，`T=109` 以内还能忍，是教学实现。

300 步训练后的输出（1.6M 参数、`d_model=8`）：

```
The key to building rapport is, to the is of.
 to your a a it and and you, in:
 and This and By points the,: customers and solution,,, customer to. By, of solutions. and
```

学会了：词形、标点、大小写、`and/the/of/to` 的搭配节奏。
学不会：语义、句子结构 —— 8 维隐层的容量天花板就在这。

## 7. 四个静默失败的坑

### 7.1 标签没有右移一位

```python
y = x          # 错：模型被要求「复述当前 token」，loss 会降到比正常更低
y = ids[i+1:i+1+T]   # 对
```

现象是 loss 掉得异常顺利，但生成时只会重复输入。判据：**训练早期 loss 是否停在 `ln(V)`**，
以及生成时是否只会复读 prompt。

### 7.2 忘了给 mask

见 [causal_mask.md](./causal_mask.md)：loss 分辨不出来，要用「改未来、前文不变」验证。

### 7.3 评估时忘了 `model.train()`

`estimate_loss` 里 `model.eval()` 之后没有切回来，后续训练的 dropout 全关掉，
loss 会突然下降一截 —— 看着像"突破了"，其实是正则没了。

### 7.4 `block_size` 超了 `max_len`

位置编码断言会报 `序列长度 X 超过位置编码上限 109`。改 `block_size`，
或者把 `transformm/__init__.py` 的 `max_len` 调大（PE 表是按 `max_len` 预生成的，不会自动变长）。

## 8. 排查清单

```python
# 1. 初始化 loss 是否 ≈ ln(V)
math.log(vocab_size)                       # 11.516

# 2. 一条样本的 x/y 是否真的错位一位
decode(x[0][:8].tolist()), decode(y[0][:8].tolist())

# 3. 梯度是否覆盖所有参数（embedding 最容易被漏掉）
[n for n, p in model.named_parameters() if p.grad is None]

# 4. 评估前后 dropout 是否恢复
model.training                             # 应为 True

# 5. 生成是否只在最后一个位置上采样
logits = model(x)                          # 只应使用 logits[:, -1, :]
```

## 心法

1. **一条 `[B,T]` 样本 = T 个训练信号**：标签是输入右移一位，别写成同一个张量。
2. **`ln(V)` 是唯一可靠的初始化基线**：初始 loss 偏离它，先去查词表维度，别急着调 lr。
3. **`eval()` 和 `no_grad()` 是两件事**：评估两个都要，训练结束记得切回 `train()`。
4. **train 降、val 平就是过拟合信号**：小模型上它来得比你想的早（本仓库 150 步左右）。
5. **能训起来 ≠ 训对了**：mask、标签错位、dropout 状态都属于"loss 照降但模型学废"，必须配结构性断言。

> 相关笔记：[causal_mask.md](./causal_mask.md)、[module_ownership.md](./module_ownership.md)（参数归属决定是否可训练）、
> [tokenizer_whitespace.md](./tokenizer_whitespace.md)（训练/推理预处理必须一致）、
> [checkpoint.md](./checkpoint.md)（训完怎么存、下次怎么接着跑）。
> 对应代码：[train.py](../transformm/train.py)、[generate.py](../transformm/generate.py)、[main.py](../transformm/main.py)、[`transformm/__init__.py`](../transformm/__init__.py)（超参集中管理）。
