import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Optional
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class KANLinear(nn.Module):
    """
    KAN线性层，使用B样条基函数实现可学习激活函数
    """
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3, grid_range: List[float] = [-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # 创建网格点
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size + 1)
        self.register_buffer('grid', grid)
        
        # B样条系数参数
        self.coefficients = nn.Parameter(torch.randn(out_features, in_features, grid_size + spline_order))
        
        # 基础权重和偏置
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.coefficients, std=0.1)
        fan_in = self.in_features
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """计算B样条基函数"""
        x = x.unsqueeze(-1)  # [batch_size, in_features, 1]
        grid = self.grid.unsqueeze(0).unsqueeze(0)  # [1, 1, grid_size + 1]
        
        # 计算B样条基函数（简化版本）
        x_scaled = (x - grid[..., :-1]) / (grid[..., 1:] - grid[..., :-1] + 1e-8)
        basis = torch.zeros_like(x_scaled)
        basis[x_scaled >= 0] = x_scaled[x_scaled >= 0]
        basis[x_scaled >= 1] = 0
        
        return basis
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基础线性变换
        base_output = F.linear(x, self.base_weight, self.bias)
        
        # KAN变换
        basis = self.b_splines(x)  # [batch_size, in_features, grid_size]
        
        # 计算样条输出
        spline_output = torch.einsum('bik,oik->bo', basis, self.coefficients[:, :, :self.grid_size])
        
        return base_output + spline_output

class MambaBlock(nn.Module):
    """
    简化的Mamba风格块，使用稳定的Transformer架构
    避免复杂的状态空间模型，但保持相似的功能
    """
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.d_inner = int(expand * dim)
        
        # 输入投影
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        
        # 深度卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # 激活函数
        self.act = nn.SiLU()
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(0.0)
        
        # 归一化
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, length, dim)
        Returns:
            (batch, length, dim)
        """
        batch, seqlen, dim = x.shape
        
        # 残差连接
        residual = x
        x = self.norm(x)
        
        # 输入投影：x -> (x1, x2)
        xz = self.in_proj(x)  # (batch, seqlen, d_inner * 2)
        x, z = xz.chunk(2, dim=-1)  # 每个都是 (batch, seqlen, d_inner)
        
        # 应用门控
        z = self.act(z)
        
        # 1D卷积
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[..., :seqlen]  # 处理padding
        x = rearrange(x, "b d l -> b l d")
        
        # 激活
        x = self.act(x)
        
        # 门控融合
        y = x * z
        
        # 输出投影
        output = self.out_proj(y)
        output = self.dropout(output)
        
        # 残差连接
        output = output + residual
        
        return output

class VisionMambaEncoder(nn.Module):
    """
    轻量化的Vision Mamba编码器
    """
    def __init__(self, dim=384, num_layers=12, num_heads=6, image_size=256, patch_size=16):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim))
        
        self.layers = nn.ModuleList([
            MambaBlock(dim=dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = x + self.pos_embed
        
        for layer in self.layers:
            x = layer(x)
        
        return self.norm(x)

class CrossViewAlignmentModule(nn.Module):
    """
    跨视角对齐模块，使用KAN增强的注意力机制
    """
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # KAN增强的查询、键、值投影
        self.q_kan = KANLinear(dim, dim)
        self.k_kan = KANLinear(dim, dim)
        self.v_kan = KANLinear(dim, dim)
        
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        B, L, D = query.shape
        
        # KAN增强的特征变换
        q = self.q_kan(query).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_kan(key).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_kan(value).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        
        return self.proj(attn_output)

class VMambaKANFSRA(nn.Module):
    """
    轻量化、对齐官方FSRA参数量的VMamba-KAN模型
    """
    def __init__(self, num_classes: int, block: int = 1, backbone: str = 'VIT-S'):
        super().__init__()
        
        # 参数映射
        if backbone == 'VIT-S':
            embed_dim = 384
            depth = 12
            num_heads = 6
        elif backbone == 'VIT-B':
            embed_dim = 768
            depth = 12
            num_heads = 12
        else: # 默认使用轻量级
            embed_dim = 256
            depth = 8
            num_heads = 4
        
        self.backbone = VisionMambaEncoder(
            dim=embed_dim,
            num_layers=depth,
            num_heads=num_heads,
            # 其他参数使用默认值
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * block, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.block = block

    def forward(self, x):
        x = self.backbone(x)
        
        # 官方实现中的分块池化
        features = []
        for i in range(self.block):
            start = i * (x.size(1) // self.block)
            end = (i + 1) * (x.size(1) // self.block)
            features.append(x[:, start:end, :].mean(dim=1))
        
        feat = torch.cat(features, dim=1)
        logits = self.classifier(feat)
        return logits, feat

# VisionMambaEncoder 和其他依赖组件需要保留或相应简化
class VisionMambaEncoder(nn.Module):
    # ... (保持之前的简化实现或进一步轻量化) ...
    pass

def create_model(num_classes: int = 701, **kwargs) -> VMambaKANFSRA:
    """创建VMamba-KAN-FSRA模型"""
    return VMambaKANFSRA(num_classes=num_classes, **kwargs)

class FSRAVMambaKAN(VMambaKANFSRA):
    """兼容旧命名：FSRAVMambaKAN 继承 VMambaKANFSRA"""
    pass

if __name__ == "__main__":
    # 测试模型
    model = create_model()
    
    # 创建假数据
    sat_img = torch.randn(2, 3, 256, 256)
    drone_img = torch.randn(2, 3, 256, 256)
    
    # 前向传播
    sat_logits, drone_logits, sat_feat, drone_feat = model(sat_img, drone_img)
    
    print(f"Satellite logits shape: {sat_logits.shape}")
    print(f"Drone logits shape: {drone_logits.shape}")
    print(f"Satellite features shape: {sat_feat.shape}")
    print(f"Drone features shape: {drone_feat.shape}")
    print("Model test passed!") 