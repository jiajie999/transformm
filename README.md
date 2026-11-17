# transformm

从零手写 mini GPT（nanoGPT 风格，不调 `nn.Transformer`），逐组件拆解 + 实测数字验证。
配套学习笔记在 [docs/](./docs/README.md)。

## 结构

```
__init__.py          全局超参（d_model / max_len / vocab_size / dropout / eps）
do_tokenize.py       cl100k_base 编解码
do_embedding.py      token → embedding
do_pos_encoding.py   正弦位置编码（buffer）
do_attention.py      单头 / 多头注意力（含因果 mask）
do_ffn.py            逐位置 FFN
do_LayerNorm.py      手写 LayerNorm
transformerBlock.py  Pre-LN + 两次残差
main.py              MiniGPT 组装 + 形状/因果性自测
train.py             next-token 训练 + checkpoint + 采样
generate.py          加载 checkpoint 续写
data/                训练语料（纯文本）
docs/                学习笔记
checkpoints/         训练产物（*.pt 已 gitignore）
```

## 快速开始

```bash
pip install -r requirements.txt

cd ..                       # 必须在仓库的父目录执行，包名是 transformm
python -m transformm.main   # 前向演示 + 自测
python -m transformm.train  # 训练（默认 300 步）
python -m transformm.generate --prompt "Customers buy from" --tokens 80
```

## 文档

| 文档 | 一句话 |
|------|--------|
| [docs/torch_vs_python.md](./docs/torch_vs_python.md) | torch 基础语法：`@`、transpose、stride、广播、掩码 |
| [docs/tokenizer_whitespace.md](./docs/tokenizer_whitespace.md) | 同一个单词为什么有两个 token id |
| [docs/causal_mask.md](./docs/causal_mask.md) | mask 必须在 softmax 之前 |
| [docs/residual_and_layernorm.md](./docs/residual_and_layernorm.md) | 残差基准与 LN 的 eps |
| [docs/module_ownership.md](./docs/module_ownership.md) | 参数 / buffer / 局部变量 |
| [docs/training_loop.md](./docs/training_loop.md) | 标签右移 + mask |
| [docs/checkpoint.md](./docs/checkpoint.md) | 权重之外还要存动量、步数、RNG |

## 训练配置

d_model 128 / 4 头 / 2 层 / block_size 32 / 1.6M 参数，CPU 约 42 秒跑 300 步，
loss 从 `ln(V)=11.516`（乱猜基线）降到 ~6.5。

## License

[MIT](./LICENSE)
