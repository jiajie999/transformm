# checkpoint：只存权重等于白存一半

> 起因：训练 300 步要 42 秒，每次实验都从头跑一遍太浪费，于是加了 checkpoint。
> 第一版只存了 `model.state_dict()`，续训时优化器是新的 —— 看起来能跑，
> 但动量是冷的，前 20 步的更新方向一直是偏的（实测低估了 0.06 的 loss 差距）。
> 这篇记清楚：一个能"接着跑"的 checkpoint 到底该装什么。

## 1. 一个 checkpoint 里装什么

```python
torch.save({
    "model": model.state_dict(),          # 参数 + 位置编码 buffer
    "optimizer": optimizer.state_dict(),  # 动量 + 已训练的步数
    "step": step,
    "config": model_config(),             # 5 个架构超参
    "args": vars(args),                   # lr / batch / block 等训练参数
    "losses": losses,                     # {"train": ..., "val": ...}
    "rng_state": torch.get_rng_state(),
}, path)
```

| 键 | 不存会怎样 |
|---|---|
| `model` | 没东西可加载 |
| `optimizer` | 续训时动量清零，更新方向偏（见 §2） |
| `step` | 续训起点未知，`--steps` 无法判断 |
| `config` | 重建模型只能靠人记，改了超参就静默错位 |
| `args` | 复现实验时不知道当时用的 lr / block_size |
| `losses` | 只知道训过，不知道训成什么样 |
| `rng_state` | 续训拿到的是另一批随机 batch（见 §6） |

实测体积（1.6M 参数、`d_model=8`）：

```
checkpoint 总大小: 19.5 MB
只存 state_dict:   6.5 MB      # 149 个 key
只存 optimizer:   13.0 MB      # 148 组动量，是模型的两倍
```

优化器比模型大一倍，是因为 Adam 每个参数要额外存 `exp_avg` 和 `exp_avg_sq` 两个同形状张量：

```python
sorted(ckpt["optimizer"]["state"][0].keys())   # ['exp_avg', 'exp_avg_sq', 'step']
```

## 2. 为什么 optimizer 必须存

AdamW 的更新不是 `w -= lr * grad`，而是先由一阶动量 `exp_avg` 和二阶动量 `exp_avg_sq` 估出方向与步长：

```
w -= lr * m_hat / (sqrt(v_hat) + eps)
```

`m`、`v` 是靠前面所有步累积出来的。新优化器从 `m=v=0` 开始，等于**重新热机**。

实测：同一份权重、同一批数据（RNG 也对齐），只差"有没有恢复动量"，续训 20 步：

```
warm-up 末期 loss: 11.07
续训前 5 步 | 完整恢复:    [11.059, 11.001, 10.978, 10.909, 10.905]
续训前 5 步 | 只恢复权重:  [11.059, 11.016, 10.990, 10.924, 10.934]
第 20 步（合计 40）| 完整: 10.281 | 只权重: 10.341
同一批数据下的 loss 差: 第1步 0.000 | 第5步 0.029 | 第20步 0.060
```

差距不是"跳变"，而是**每天都在偏**：第 1 步看不出，第 20 步已经差 0.06。
短程续训不明显，长时间续训会一直吃亏 —— 这也是为什么它很难被察觉。

## 3. 为什么不存整个模型对象

`torch.save(model)` 走 pickle，记录的是 `transformm.main.MiniGPT` 这个**类路径**。
换目录、改类名、改构造参数，都会在读的时候炸；而且 pickle 能执行任意代码。

所以存 `state_dict()`（纯张量）+ 基础类型（int / float / str / dict / list）。
好处是能用 `weights_only=True` 安全读取，坏处是**不能塞复杂对象**：

```python
torch.save({"args": argparse.Namespace(lr=1e-3)}, "ns.pt")
torch.load("ns.pt", weights_only=True)
# WeightsUnpickler error: Unsupported global: GLOBAL argparse.Namespace was not an
# allowed global by default. ...
```

这就是 `"args": vars(args)` 而不是 `"args": args` 的原因 —— `vars()` 把 Namespace 变成普通 dict。

```python
ckpt = torch.load(path, map_location=map_location, weights_only=True)
```

`map_location="cpu"` 则保证"在有 GPU 的机器上训练、在笔记本上生成"不会因为 device 不匹配而失败。

## 4. 三种读不出来的情况

**① 文件不存在**（最常见：忘了先训练）

```
AssertionError: 找不到 checkpoint /tmp/cktest/nope.pt，先跑一次 python -m transformm.train
```

**② 架构超参不一致**（改了 `d_model` / `num_blocks` / `vocab_size`）

```
AssertionError: checkpoint 与当前配置不一致 (saved, current): {'d_model': (16, 8)}
```

所以 checkpoint 里要存 `config`，加载时**先比超参、再载权重**。
如果只靠 `load_state_dict` 报错，你会看到一堆 key 名字，而不知道根因是超参变了。

**③ key 对不上**（改成 strict 加载时的原始报错）

```
RuntimeError: Error(s) in loading state_dict for MiniGPT:
	Missing key(s) in state_dict: "blocks.0.LN1.gamma".
	Unexpected key(s) in state_dict: "blocks.0.bogus".
```

`load_state_dict` 默认 `strict=True`，少 key 或多 key 都报错 —— **这是好事**。
`strict=False` 会静默半加载，模型带着随机权重继续训练，loss 卡在某个值下不去，极难排查。

## 5. 哪些东西**不该**进 checkpoint

`do_attention.py` 里的因果 mask 是 `persistent=False` 的 buffer：

```python
self.register_buffer("causal_mask", ..., persistent=False)
```

它不在 `state_dict` 里（149 = 148 参数 + 1 个 `pos_encoding.pe`），加载时按 `max_len` 重建。
这样即使以后把 `max_len` 从 109 调大，旧 checkpoint 也不会报缺 key。

判断标准：**能从超参推出来的张量就不存**。位置编码表恰好相反 —— 它是可替换的常量
（换成学出来的 PE 就要存），所以留在 `state_dict` 里。详见 [module_ownership.md](./module_ownership.md)。

## 6. 续训要做到"接得上"：逐位验证

光"能加载"不够，要证明"续训 = 没中断过"。做法是跑两条路径对比：

```bash
# A：一口气跑 40 步
python -m transformm.train --steps 40 --ckpt a.pt --eval-interval 1000

# B：跑 20 步，再续 20 步
python -m transformm.train --steps 20 --ckpt b.pt --eval-interval 1000
python -m transformm.train --steps 40 --resume --ckpt b.pt --eval-interval 1000
```

结果（A 与 B 的最终 checkpoint 对比）：

```
model 权重逐位相同: True | key 数: 149
optimizer.state 条目数: 148 | lr 0.003 | betas (0.9, 0.95)
exp_avg 逐位相同: True
rng_state 相同: True
最终 train 10.216 | val 10.322   ← 两条路径数字完全一致
```

能做到逐位相同，靠的是三件事：**权重 + 动量 + RNG** 都存了。少任何一个都会分叉。

### 为此改的一处：评估不再污染 RNG

`estimate_loss` 会抽随机 batch，也就是消费全局 RNG。原来它不管进出，导致
"评估次数不同 → 后续 batch 序列不同 → 续训结果不同"。

```python
rng_state = torch.get_rng_state()
model.eval()
...
torch.set_rng_state(rng_state)
```

副作用（好的那种）：**300 步的曲线现在只由 seed 决定，与 `--eval-interval` 无关**，
跑两次数字完全一致（6.033 / 6.521）。之前因为评估会偷走 RNG，同等配置换个
`eval-interval` 就会得到另一条曲线（旧记录是 5.932 / 6.661，就是被这个问题污染的）。

### 顺带修的一个静默失效：`--lr` 会被 checkpoint 盖掉

```python
o = torch.optim.AdamW(model.parameters(), lr=0.01)
o.load_state_dict(ckpt["optimizer"])
o.param_groups[0]["lr"]     # 0.003 —— 构造时给的 0.01 被旧值覆盖了
```

`optimizer.state_dict()` 里含 `param_groups`（lr、betas、weight_decay），加载时整组覆盖。
所以“续训时把 lr 调小一点”这条常见操作会**完全不生效，而且不报错**。

修法是加载后显式写回，并把差异打印出来：

```python
for group in optimizer.param_groups:
    group["lr"] = args.lr
```

```
  注意: checkpoint 里的 lr=0.003，本次 --lr=0.01，以本次为准
```

## 7. 日常用法

```bash
# 训练：每 100 步 + 结束时各存一次，默认落到 checkpoints/last.pt
python -m transformm.train

# 中途被打断 / 想多训 200 步
python -m transformm.train --resume --steps 500

# 换个文件名，保留旧结果（--ckpt 是存与续训共用的路径）
python -m transformm.train --steps 600 --ckpt checkpoints/run2.pt

# 加载权重直接续写，不读语料、不建优化器
python -m transformm.generate --prompt "Customers buy from" --tokens 80
```

生成脚本的输出（300 步模型）：

```
checkpoint: checkpoints/last.pt
step 300 | 参数量 1614528 | 保存时 loss {'train': 6.033, 'val': 6.521}
采样: temperature=0.8 | top_k=50 | seed=0

prompt: 'The key to building rapport is'
The key to building rapport is. also the a is. By or potential,. By in the This.
the approach, the you and By it solutions, you and pain also a it of.
```

`generate.py` 里还有一条断言：**同一 seed 生成两次必须逐字相同**。
它顺便验证了模型确实在 `eval()` 模式 —— dropout 一旦生效，两次结果必然不同。
（这也暴露过一个 bug：`generate()` 原来结尾无条件 `model.train()`，推理脚本调用一次就把模型切回训练模式了。）

## 8. 排查清单

```python
# 1. checkpoint 里有什么
ckpt = torch.load(path, weights_only=True); print(sorted(ckpt))

# 2. 存了动量没有（没有就是白存一半）
"optimizer" in ckpt and len(ckpt["optimizer"]["state"])

# 3. 架构超参对不对
ckpt["config"]                      # 与 transformm/__init__.py 比

# 4. 权重是否真的完整加载
model.load_state_dict(ckpt["model"])  # strict=True，报错就说明有 key 不匹配

# 5. 续训是否接得上
torch.equal(torch.load(a)["model"]["embedding.weight"],
            torch.load(b)["model"]["embedding.weight"])
```

## 心法

1. **能"接着跑"的 checkpoint = 权重 + 优化器状态 + 步数 + 超参 + RNG**，缺一样就接不回来。
2. **Adam 的动量是半条命**：省掉 13 MB 的优化器，换来一条一直在偏的续训曲线。
3. **存 `state_dict`，不存对象**：避免 pickle 的类路径绑定，才敢用 `weights_only=True` 读陌生文件。
4. **加载前先比超参，再载权重**：`config` 让"改了 d_model"变成一个说得清的断言，而不是一堆 missing keys。
5. **评估要恢复 RNG**：否则"训到哪一步"之外，"评估了几次"也会改变结果 —— 实验就不可复现了。

> 相关笔记：[training_loop.md](./training_loop.md)（被 checkpoint 保存的那个循环）、
> [module_ownership.md](./module_ownership.md)（`state_dict` 里为什么是 149 个 key）。
> 对应代码：[train.py](../transformm/train.py)、[generate.py](../transformm/generate.py)、[../checkpoints/](../checkpoints/)。
