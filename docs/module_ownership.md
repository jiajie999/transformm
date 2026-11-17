# 参数归属：为什么 embedding 一挪出模型，训练就废了

> 起因：为了复用 `get_pe_eb()`，把 embedding 查表和位置编码加在一起、放在模型外面。
> 直觉上"模型还是那个模型"，实际上参数量从 1,614,528 掉到 812,312，`state_dict` 里没了 embedding，
> 同一句话两次前向拿到完全不同的结果。
> 结论：**一个张量"归谁所有"决定了三件事 —— 能不能训练、能不能保存、能不能复现**。
> 这三条才是判断模块设计对不对的标准，不是"代码看起来更短"。

## 1. 三种东西，三种归宿

| 东西 | 例子 | 该放哪 | 在 `parameters()` | 在 `state_dict()` |
|------|------|--------|------------------|------------------|
| 可学习参数 | `nn.Embedding` / `nn.Linear` 的权重 | 模块成员（`self.xxx`） | ✓ | ✓ |
| 常量表（要保存/搬家） | 正弦位置编码 | `register_buffer` | ✗ | ✓ |
| 常量表（可推导） | 因果 mask | `register_buffer(persistent=False)` | ✗ | ✗ |
| 临时张量 | 函数里新建的 `nn.Embedding` | 局部变量 | ✗ | ✗ |

实测 MiniGPT 的账本（`d_model=8, num_blocks=12`）：

```
state_dict 键数: 149 | parameters 数: 148
差集（只在 state_dict 里）: ['pos_encoding.pe']
每层 block 的键数: 12        →  12 × 12 + embedding 1 + ln_final 2 + lm_head 1 = 148
```

```
参数量 1614528 = embedding 802216 + lm_head 802216 + blocks 10080 + ln_final 16
```

两个 802,216 就是 `100277 × 8`：embedding 和 lm_head 的形状互为转置
（`V×C` 与 `C×V`），占了模型的 99.4%。**把它们之一挪出模型，等于训练半个模型。**

## 2. 三条判据（可训练 / 可保存 / 可复现）

| 判据 | 检查方式 | embedding 在模型内 | embedding 在 `get_pe_eb` 里 |
|------|---------|-----------------|--------------------------|
| 能训练 | 在 `parameters()` 里 | ✓ | ✗ |
| 能保存 | 在 `state_dict()` 里 | ✓ | ✗ |
| 能复现 | 同输入两次前向 `torch.equal` | ✓ | ✗ |

"能复现"这一条最容易被忽略，因为它不报错。实测同一句 `"猫吃鱼"`：

```
embedding 在模型里  → 两次运行 top-5 都是 [65948, 17614, 53459, 55374, 87270]
embedding 在 helper 里 → 第一次 [40832, ...]，第二次 [65948, ...]
```

原因是 `embedding_build()` 每次调用都新建 `nn.Embedding(vocab_size, d_model_len)`，
权重是重新随机初始化的。`torch.manual_seed(42)` 只固定了**建立时序**上的初始化，
固定不了"调用时再建一次"的表。

这三条可以直接写成断言，`main.py` 的 `__main__` 里就是这么做的：

```python
params = dict(model.named_parameters())
assert "embedding.weight" in params                                   # 能训练
assert "pos_encoding.pe" in model.state_dict()                        # 能保存（buffer）
assert "pos_encoding.pe" not in params                                # 但不该被训练
with torch.no_grad():
    assert torch.equal(model(ids), logits)                            # 能复现
```

## 3. `Parameter` 与 `buffer` 的差别就一行

```python
self.gamma = nn.Parameter(torch.ones(d_model))            # 参数：需要梯度
self.register_buffer("pe", pos_encode(d_model, max_len))  # buffer：常量但跟着模型走
```

| | `Parameter` | `buffer` | 局部变量 |
|---|---|---|---|
| `parameters()` | ✓ | ✗ | ✗ |
| `state_dict()` | ✓ | ✓ | ✗ |
| `.to(device)` 一起搬家 | ✓ | ✓ | ✗ |
| 优化器更新 | ✓ | ✗ | ✗ |
| `requires_grad` | ✓ | ✗ | ✗ |

判断标准很简单：**"这个值需要被梯度改吗？"** 需要 → `Parameter`；不需要但必须跟着模型走
（保存、搬到 GPU） → `buffer`；两者都不需要 → 局部变量。

`pos_encode()` 是纯函数（同样输入永远同样输出），所以它的产物天然属于 buffer 这一类。

## 4. 第三种 buffer：`persistent=False`

因果 mask 也是常量，但它能从 `max_len` 完全推导出来，**不该进 checkpoint**：

```python
self.register_buffer("causal_mask", torch.triu(...), persistent=False)
```

```python
m = MultiHeadAttention(num_heads=4, d_model_len=8)
any("causal_mask" in k for k in m.state_dict())      # False
```

如果让它进 `state_dict`，把 `max_len` 从 109 改成 256 再加载旧权重时会报
`Missing key(s) in state_dict: "...causal_mask"` —— 一个根本不需要保存的东西拦住了加载。

## 5. 注册：`ModuleList` 与普通 `list` 的差别

```python
self.blocks = [Block() for _ in range(12)]         # ❌ 什么都没注册
self.blocks = nn.ModuleList([Block() for _ in range(12)])   # ✓
```

普通 list 里放模块，Python 层面能跑，但 PyTorch 看不见它们：

| | `list` | `ModuleList` |
|---|---|---|
| `parameters()` | 看不到 | ✓ |
| `state_dict()` | 看不到 | ✓（键 `blocks.0.…`） |
| `.to(device)` / `.eval()` | 不生效 | ✓ |

这类错误在 `main.py` 里用一句断言锁住：

```python
assert all(any(key.startswith(f"blocks.{i}.") for key in model.state_dict())
           for i in range(num_blocks)), "有 block 没被注册进 state_dict"
```

比 `assert len(list(model.parameters())) > 0` 强得多 —— 后者在"只有 12 层里的 1 层被注册"时也照样通过。

## 6. 重复注册：同一个层出现两次

`nn.Sequential` 里再塞一个已经命名的属性，同一个参数会被注册两遍：

```python
s.a = nn.Linear(2, 2)
s.s = nn.Sequential(s.a)          # 同一层又包了一层
```

实测：

```
state_dict 键: ['a.weight', 'a.bias', 's.0.weight', 's.0.bias']     ← 4 个键，2 份权重
parameters 数: 2 (去重后)                                            ← 优化器只看到 2 个
```

后果：

- checkpoint 体积翻倍（同一份权重存两次）；
- 优化器**不会**更新两遍（`parameters()` 会去重），但 `state_dict` 里两份永远一致，
  看起来"没问题"，直到你试图单独加载其中一部分；
- 想冻结/替换其中一份时，另一份还在生效。

**修法**：要么用命名属性 + 显式 forward，要么全用 `nn.Sequential` 并只从序号访问，不要混着来。
本仓库的 `FFN` 就是显式写的：

```python
self.linear1 = nn.Linear(d_model, hidden)
self.linear2 = nn.Linear(hidden, d_model)
self.dropout = nn.Dropout(dropout_rate)
self.ReLU = nn.ReLU()

def forward(self, x):
    return self.linear2(self.dropout(self.ReLU(self.linear1(x))))
```

## 7. `get_pe_eb()` 的定位：脚手架，不是生产路径

这套代码是学习用的，"一次跑通看形状"很重要，所以保留 helper：

```python
def get_pe_eb(text=None):
    """文本 → embedding + 位置编码，[B,T,C]。仅用于学习/调试。"""
    emb = embedding_build(text)
    return PositionalEncoding(d_model=emb.size(-1))(emb)
```

但**生产路径必须是 `pos(emb(ids))`**，两个类各司其职：

```python
x = self.embedding(token_ids)      # nn.Embedding：参数
x = self.pos_encoding(x)           # PositionalEncoding：只做加法
```

这次改动走了个来回：先按"复用 helper"把 embedding 挪出去（参数量掉一半），
发现训练/复现全废，再改回来。留下的判据是 [train.py](../train.py) 能被
`optimizer = AdamW(model.parameters())` 完整覆盖，且反向传播能走到每一个参数：

```python
model(token_ids).mean().backward()
assert not [n for n, p in model.named_parameters() if p.grad is None]
assert model.embedding.weight.grad.abs().sum() > 0
assert model.embedding.weight.grad[pad_token].abs().sum() == 0    # padding_idx 那行不更新
```

## 8. 排查清单

```python
# 1. 关键参数在不在（embedding / lm_head 最容易丢）
[k for k in model.state_dict() if "embedding" in k or "head" in k]

# 2. 参数量对不对得上（手算一遍）
sum(p.numel() for p in model.parameters())

# 3. 有没有"只在 state_dict 里"的东西（应该是 buffer）
set(model.state_dict()) - set(dict(model.named_parameters()))

# 4. 有没有参数拿不到梯度（漏注册 / 断链）
[n for n, p in model.named_parameters() if p.grad is None]

# 5. 同一输入两次前向是否一致（暴露"调用时新建表"）
torch.equal(model(ids), model(ids))
```

## 心法

1. **先问"归谁所有"，再看"代码短不短"**：参数必须在模块成员上，否则优化器看不见。
2. **三条判据背下来**：`parameters()` 管训练、`state_dict()` 管保存、`torch.equal` 管复现。
3. **`Parameter` 会被梯度改，`buffer` 只跟着走**：判断标准是"这个值需要学吗"。
4. **能推导出来的常量用 `persistent=False`**：别让无关的 key 破坏 checkpoint 兼容性。
5. **`ModuleList` 是注册，`list` 不是**；`Sequential` 和命名属性混用会重复注册，checkpoint 悄悄翻倍。
6. **学习用 helper 可以，但要对它标注"非生产路径"**：`get_pe_eb` 存在的意义是看形状，不是凑流程。

> 相关笔记：[causal_mask.md](./causal_mask.md)（`persistent=False` 的 buffer）、
> [training_loop.md](./training_loop.md)（参数归属直接决定训练能不能跑）、
> [tokenizer_whitespace.md](./tokenizer_whitespace.md)（`do_embedding.py` 里那张表为什么每次都不一样）。
> 对应代码：[main.py](../main.py)、[do_pos_encoding.py](../do_pos_encoding.py)、[do_embedding.py](../do_embedding.py)。
