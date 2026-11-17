# cl100k_base 的空格：同一个单词为什么有两个 token

> 起因：`transformm/do_pos_encoding.py` 里把 `"cat eat fish"` / `"fish eat cat"` 两行送进 `nn.Embedding`，
> 发现 `cat` 在两行里的向量**不一样**。
> 结论：不是 embedding 层的问题，是 tokenizer 把 `cat` 和 ` cat` 切成了两个不同的 token id。
> 这是 GPT 系列的**既定设计**，不影响训练正确性，不需要「修」。

## 1. 现象与证据

实测 `transformm/do_pos_encoding.py` 的输入：

```
line1 "cat eat fish"  → [4719, 8343, 7795]
line2 "fish eat cat"  → [18668, 8343, 8415]

4719  = 'cat'     # 无前导空格
8415  = ' cat'    # 带一个前导空格
18668 = 'fish'
7795  = ' fish'
8343  = ' eat'    # 两行中间位置，完全一致
```

`nn.Embedding` 本质是「按 token id 查行」的表，无状态：**同一个 id 必然查到同一个向量**。
所以两行里 `eat` 向量完全相同（都是 8343），只有 `cat` 不同 —— 因为 id 本来就不同。

## 2. 根因：正则把前导空白吸进了切分单元

`cl100k_base` 的预切分正则里有一段 `[^\r\n\p{L}\p{N}]?\p{L}+`，含义是
「**至多一个**非字母/数字/换行字符 + 一串字母」。那个可选字符通常就是空格。

于是：

| 词出现位置 | 切分单元 | token |
|-----------|---------|-------|
| 行首 / 换行后 / 引号后 | `cat` | `cat` (4719) |
| 两词之间 | ` cat` | ` cat` (8415) |

`line.strip()` 只去掉整行首尾空白，**帮不上忙** —— 差异来自词与词之间那个空格本身，属于正常上下文。

自测一行即可确认：

```python
encoding.encode("cat is here")   # [4719, ...]   词首形态
encoding.encode(" cat is here")  # [8415, ...]   空格形态
```

## 3. 为什么不影响训练正确性

在 `data/sales_textbook.txt`（77,919 tokens）上统计：

```
total 77919   leading-space tokens 64862 (83.2%)   distinct ids 3771
```

83% 的 token 都是「带前导空格」形态，它是**主流路径**而不是边角情况。而且：

- **它携带真实信息**：模型能区分「这是一个新词的词首」和「这个词跟前面的词粘在一起」，这个边界线索有助于预测下一个 token。
- **跨词边界不会被错误合并**：正则切分保证 `cat fish` 不会 merge 成 `catfish`。
- **不增加词表**：两种形态共用同一张 100,277 行的表，只是更多行被用到。
- **训练/推理天然一致**：只要两侧用同一个 tokenizer 实例（`transformm/do_tokenize.py` 里模块级 `encoding` 已满足）。

## 4. 真正会伤训练结果的四件事

空格本身不是问题，下面这些才是：

### 4.1 训练与推理的预处理不一致

`scifi-demo/train-scifi.py` 用 `encoding.encode(text)` 编码整个文件 —— 正确，空白是天然的。
但 `scifi-demo/inference-scifi.py` 对裸 prompt 直接 `encoding.encode(start)`：

```python
encoding.encode("cat is here")   # 4719 → 词首形态
encoding.encode(" cat is here")  # 8415 → 训练语料里 83% 的形态
```

**续写型 prompt** 的首 token 应该用空格形态，否则第一个位置走的是训练中几乎没出现过的 embedding 行。
便宜的修法：`encoding.encode(" " + start)`，或让 prompt 保持与语料中完全一致的写法。

### 4.2 只有一侧 `.strip()`

`transformm/do_embedding.py` 的 `line.strip()` 对 demo 没问题，但不能「训练读 strip 过的行、推理读原始行」。
稳妥做法：token id **只生成一次**并缓存到磁盘，从根上杜绝 train/serve 分叉。

### 4.3 小语料 + 大词表（对本仓库最现实的一条）

`sales_textbook.txt` 只用到 100,277 行里的 **3,771** 行。
`do_embedding.py` 里 `nn.Embedding(100277, 8)` 约 80 万参数，其中 **约 96% 永远是随机初始化值**，
等于建了一张 96% 是噪声的表。想让 demo 有教学意义，就按语料自建词表：

```python
ids = encoding.encode(text)
vocab = sorted(set(ids))
tok2id = {t: i for i, t in enumerate(vocab)}
emb_layer = nn.Embedding(len(vocab), d_model_len)
```

### 4.4 冻结/LoRA 时梯度被摊薄

两种形态意味着每个词形只拿到一半的更新量。真实规模下可忽略，但小模型上会明显拖慢收敛。

## 5. `nn.Embedding` 每次调用都重新随机初始化

`do_embedding.py` 的表是**函数内局部变量**，每次 `embedding_build()` 都新建一次，权重都不同。

- 单次调用内对比同一张表：公平（上面所有结论都成立）。
- 想跨调用/跨运行比较向量：需要模块级建表，或 `torch.manual_seed(0)` 固定。

## 6. 排查清单

```python
# 1. 确认两个词到底是不是同一个 id
encoding.encode("cat eat fish")   # [4719, 8343, 7795]
encoding.encode("fish eat cat")   # [18668, 8343, 8415]

# 2. 确认「同 id → 同向量」
torch.allclose(emb[0, 1], emb[1, 1])   # True  两个 eat
torch.allclose(emb[0, 0], emb[1, 2])   # False cat vs ' cat'

# 3. 打印的是 emb 还是 emb + pe？（只打印 emb 会看不到位置编码的效果）
print(x)
```

## 心法

1. **同一个单词两种向量，先怀疑 tokenizer，再怀疑 embedding** —— 查 id，别查浮点数。
2. **前导空格是 token 的一部分**，`strip()` 改不了行内空白，跨行拼接时尤其要小心。
3. **照抄同一套预处理**：训练怎么切，推理就怎么切，最好只切一次并缓存。
4. **词表要跟着语料走**：语料里只有几千个 id，就不要建十万行的表。

> 相关笔记：`nn.Embedding` 该放在哪（参数 vs buffer vs 局部变量），见 [module_ownership.md](./module_ownership.md)；
> 训练/推理预处理必须一致这条在 [training_loop.md](./training_loop.md) 里又出现了一次（错位标签与滑窗）；
> 基础 torch 张量操作见 [torch_vs_python.md](./torch_vs_python.md)。
