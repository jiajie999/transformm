"""全局配置：模型尺寸与各子层的固定超参统一放这里，其它模块不再写字面量。

    from transformm import d_model_len, vocab_size

改这里的值就能整体换配置。注意这些是模块级常量，不要在函数里用同名局部变量覆盖（遮蔽）。
"""

# ---- 模型结构 ----
d_model_len = 8        # 特征维 C：所有子层输入输出的最后一维
max_len = 109          # 位置编码支持的最大序列长度
num_heads = 4          # 多头注意力的头数，必须能整除 d_model_len
num_blocks = 12        # TransformerBlock 的层数

# ---- 词表 ----
vocab_size = 100277    # cl100k_base 词表大小：embedding 的行数与 LM head 的输出维都用它
pad_token = 0          # 补齐用的 token id，同时作为 embedding 的 padding_idx

# ---- 子层超参 ----
ffn_expansion = 4      # FFN 升维倍数：d_model_len -> ffn_expansion * d_model_len
dropout_rate = 0.1     # FFN 里的 dropout 比例
ln_eps = 1e-6          # LayerNorm 的 eps，防止方差为 0 时除出 nan
