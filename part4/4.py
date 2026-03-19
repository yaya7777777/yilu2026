import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # 定义三个线性层: W_q, W_k, W_v
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, d_model]

        # 1. 生成 Q, K, V
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # 2. 计算 Attention Score = Q @ K^T / sqrt(d_model)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        # 3. 加上 Mask (可选，如果mask不为None)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # 4. Softmax + 与 V 相乘
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        

        return output, attention_weights
    
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = SingleHeadSelfAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 前馈网络通常是: Linear -> ReLU -> Linear
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x, mask=None):
        # Part 1: Attention + Add + Norm
        # 注意：这里是面试常考点，究竟是先Norm还是后Norm？(这里采用经典的 Post-LN)

        attn_out, _ = self.attention(x, mask)
        x = self.norm1(x + attn_out) # [关键代码] 残差连接: x + sublayer(x)

        # Part 2: FFN + Add + Norm
        # 请补全这部分代码，实现 FeedForward 的残差连接结构
        ffn_out = self.feed_forward(x)
        x = self.norm2(x + ffn_out)

        return x


# 测试代码
x = torch.randn(2, 10, 64)
model = TransformerBlock(d_model=64)
out = model(x)
print("shape:", out.shape)
mean = out.mean().item()
var = out.var().item()
print(f"mean:{mean:.4f}, var:{var:.4f}")