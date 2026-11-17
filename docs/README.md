# transformm 学习笔记

从组件到训练的阅读顺序（每一篇都以「实测数字 + 心法」结尾）：

| 文档 | 一句话 | 关键数字 |
|------|--------|---------|
| [torch_vs_python.md](./torch_vs_python.md) | torch 的基础语法：`@`、transpose、stride、广播、布尔掩码、dtype/相等性、注册与训练调用 | `d_k=4` 的 `[B,H,T,T]` 骨架；float32 只有 7 位有效数字 |
| [tokenizer_whitespace.md](./tokenizer_whitespace.md) | 同一个单词为什么有两个 token id | 83.2% 的 token 带前导空格 |
| [causal_mask.md](./causal_mask.md) | mask 必须在 softmax **之前**，`diagonal=1` 留出自己 | 追加 token 后前文变化：0.0 vs 0.155 |
| [residual_and_layernorm.md](./residual_and_layernorm.md) | 残差基准逐级推进；LN 用 `sqrt(var+eps)` | 注意力贡献 1.04 vs 0.22；eps 让梯度 1.5e6 → 850 |
| [module_ownership.md](./module_ownership.md) | 参数 / buffer / 局部变量决定能否训练、保存、复现 | 参数量 1,614,528 vs 812,312 |
| [training_loop.md](./training_loop.md) | 标签右移一位 + mask，其余都是调参 | loss 11.516 → 6.521（300 步 / 42 秒） |
| [checkpoint.md](./checkpoint.md) | 存的不只是权重：动量 + 步数 + RNG 都要 | 19.5 MB = 模型 6.5 + 优化器 13.0 |

## 三条贯穿全部笔记的原则

1. **形状对 ≠ 算对了。** 残差基准写错、`x - mean/std` 优先级写错、mask 顺序写反，全都形状正确、loss 照降。
   每个结论都要有数值层面的对照实验（置零、对照手写前向、对照官方实现）。
2. **先问"归谁所有"，再问"代码短不短"。** 参数必须在模块成员上（`parameters()`），
   常量进 buffer（`state_dict()`），能推导的常量 `persistent=False`。
3. **基线先算清楚。** `ln(V) = 11.516` 是"在 10 万个 token 里乱猜"的期望损失；
   初始化 loss 离开这个数，先查词表维度，别急着调 lr。

## 代码地图

| 文件 | 职责 |
|------|------|
| [`__init__.py`](../__init__.py) | 全局超参（d_model / max_len / vocab_size / dropout / eps） |
| [do_tokenize.py](../do_tokenize.py) | cl100k_base 编解码 |
| [do_embedding.py](../do_embedding.py) | 文本 → token → embedding（demo 用，表是临时的） |
| [do_pos_encoding.py](../do_pos_encoding.py) | `pos_encode()` 纯函数 + `PositionalEncoding`（buffer） |
| [do_attention.py](../do_attention.py) | 单头 `Attention`、多头 `MultiHeadAttention`（含因果 mask） |
| [do_ffn.py](../do_ffn.py) | 逐位置 FFN（升维 ×4 → ReLU → 降维） |
| [do_LayerNorm.py](../do_LayerNorm.py) | 自己实现的 LayerNorm |
| [transformerBlock.py](../transformerBlock.py) | Pre-LN + 两次残差 |
| [main.py](../main.py) | `MiniGPT` 组装 + 前向形状/因果性自测 |
| [train.py](../train.py) | next-token 训练循环 + checkpoint 存取 + 采样生成 |
| [generate.py](../generate.py) | 加载 checkpoint 直接续写（不重新训练） |
| [checkpoints/](../checkpoints/) | 训练产物（`last.pt`，已 gitignore） |

## 还没写的

- KV cache：现在生成每步重算整段，`max_len=109` 以内还能忍。
- weight tying：embedding 与 lm_head 各 802,216 参数，绑定后省一半。
- 多 checkpoint 管理：现在只有 `last.pt`，还没有 best / 按 step 存档与保留策略。
