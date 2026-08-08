#!/usr/bin/env python3
"""
低秩投影交互模型
参数量从亿级降到百万级
"""

import torch
import torch.nn as nn


class LowRankInteractionModel(nn.Module):
    """
    低秩投影交互模型

    架构:
    1. 投影: esm_dim (5120) → proj_dim (128/256)
    2. 交互: concat([v, h, v*h, |v-h|])
    3. MLP分类器
    """

    def __init__(self, esm_dim: int = 1280, proj_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.4):
        super().__init__()

        # Large structural embeddings (e.g., ESMFold joint) can have huge scale and
        # saturate downstream Sigmoid, leading to zero gradients. Normalize inputs.
        self.viral_norm = nn.LayerNorm(esm_dim)
        self.host_norm = nn.LayerNorm(esm_dim)

        # 投影层：降维
        self.viral_proj = nn.Linear(esm_dim, proj_dim)
        self.host_proj = nn.Linear(esm_dim, proj_dim)

        # 交互特征维度: v + h + v*h + |v-h| = 4 * proj_dim
        interaction_dim = 4 * proj_dim

        # MLP分类器（输出 logits；Sigmoid 在 loss/评估阶段统一处理）
        self.classifier = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, viral_emb: torch.Tensor, host_emb: torch.Tensor) -> torch.Tensor:
        viral_emb = self.viral_norm(viral_emb)
        host_emb = self.host_norm(host_emb)

        # 投影到低维空间
        v_proj = self.viral_proj(viral_emb)  # [batch, proj_dim]
        h_proj = self.host_proj(host_emb)    # [batch, proj_dim]

        # 构建交互特征
        v_h_product = v_proj * h_proj        # 逐元素乘积
        v_h_diff = torch.abs(v_proj - h_proj)  # 绝对差值

        # 拼接所有特征
        interaction = torch.cat([v_proj, h_proj, v_h_product, v_h_diff], dim=1)  # [batch, 4*proj_dim]

        # 分类
        output = self.classifier(interaction)  # [batch, 1] logits

        return output.squeeze(-1)  # [batch]


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 测试不同配置的参数量
    configs = [
        {'proj_dim': 128, 'hidden_dim': 256},
        {'proj_dim': 256, 'hidden_dim': 512},
    ]

    print("参数量对比:")
    print("-" * 60)

    for config in configs:
        model = LowRankInteractionModel(**config)
        params = count_parameters(model)
        print(f"proj_dim={config['proj_dim']}, hidden_dim={config['hidden_dim']}: {params:,} 参数 ({params/1e6:.2f}M)")

    print("-" * 60)
    print("对比: 原始双线性模型 (hidden_dim=256): 419,496,705 参数 (419.50M)")
