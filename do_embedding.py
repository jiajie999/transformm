import os
import sys

# 直接 python xxx.py 时 sys.path[0] 是本目录，`from transformm import` 会失败；把父目录加进来。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformm import d_model_len, pad_token, vocab_size
from transformm.do_tokenize import encode


def embedding_build(text):

    # with open('data/sales_textbook.txt', 'r',
    #           encoding="utf-8") as file:
    lines = [encode(line.strip()) for line in text.split("\n")]

    seq_len = max(len(t) for t in lines)
    # 右侧 padding：每条补齐到 seq_len，缺失位置填 pad_token
    tokens_to_embedding = torch.tensor([t + [pad_token] * (seq_len - len(t)) for t in lines])

    # num_embeddings 必须覆盖 cl100k_base 的最大 token id，用全局 vocab_size，不能用行数
    embedding = nn.Embedding(vocab_size, d_model_len, padding_idx=pad_token)

    # print(tokens_to_embedding.shape)          # [num_lines, seq_len]
    emb = embedding(tokens_to_embedding)      # [num_lines, seq_len, d_model_len]
    # print(emb.shape)
    # print(embedding.weight[pad_token])        # padding_idx 对应行恒为 0
    # print(embedding.weight.shape)
    return emb

if __name__ == "__main__":
    text="""Chapter 1: Building Rapport and Capturing Attention
Subpoint: Understanding the Importance of Building Rapport
Building rapport is a fundamental skill in sales that cannot be underestimated. It lays the foundation for establishing a connection with your potential customers, gaining their trust, and ultimately convincing them to make a purchase. Rapport can be defined as a harmonious relationship based on mutual understanding and empathy. When you build rapport with someone, you create a sense of familiarity, comfort, and shared interests, making it easier to communicate and influence their decision-making process.
One of the main reasons why building rapport is crucial in sales is that people are more likely to buy from someone they like and trust. By establishing a positive and genuine connection with your customers, you increase their confidence in you and your product or service. People want to do business with individuals they feel comfortable with, those who understand their needs and genuinely care about their success. When you show a sincere interest in your customers, they are more likely to reciprocate that interest, leading to a stronger rapport.
Additionally, building rapport helps to create an open and conducive environment for effective communication. When customers feel comfortable with you, they are more willing to share their needs, pain points, and concerns openly. This information is vital for understanding their unique requirements and tailoring your sales approach to address those specific needs. By actively listening and demonstrating empathy, you show your customers that you genuinely care about their challenges and are committed to finding the best solution for them.
Furthermore, building rapport allows you to differentiate yourself from competitors. In today's highly competitive market, where customers have numerous options to choose from, it is essential to stand out. By establishing a personal connection with your customers, you create a memorable experience that sets you apart. Customers are more likely to remember and recommend someone who made them feel understood and valued. In addition, building rapport enables you to establish common ground and shared interests, reinforcing the connection and making your customers feel that you truly understand their world.
Moreover, rapport-building is not just a one-time effort but an ongoing process. It involves consistently nurturing and maintaining the relationship with your customers. By staying in touch, providing relevant and valuable information, and showing continued interest, you reinforce the rapport you have built. This helps to generate repeat business and foster long-term customer loyalty.
To understand the importance of building rapport, you must recognize that sales is not just about selling a product or service; it is about building relationships. By investing time and effort in building rapport, you lay the groundwork for a successful sales journey. Building rapport establishes trust, opens channels of communication, differentiates you from competitors, and leads to long-term customer relationships. Remember, customers buy from people they like and trust, and building rapport is the key to unlocking their willingness to buy from you. Chapter 1: Building Rapport and Capturing Attention
Subpoint: Creating a Positive First Impression"""
    emb=embedding_build(text)
    print(emb)
    print(emb.shape)