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
    简化的Mamba块，用于实现状态空间模型
    """
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.dim)
        
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv-1, groups=self.d_inner)
        self.activation = nn.SiLU()
        
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, d_state, bias=True)
        
        # 状态空间参数
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        x_and_res = self.in_proj(x)  # [B, L, 2 * d_inner]
        x, res = x_and_res.split(split_size=self.d_inner, dim=-1)
        
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[..., :L]
        x = rearrange(x, 'b d l -> b l d')
        x = self.activation(x)
        
        # 选择性扫描
        x_dbl = self.x_proj(x)  # [B, L, d_state * 2]
        dt, B_proj = x_dbl.split(split_size=self.d_state, dim=-1)
        dt = self.dt_proj(x)  # [B, L, d_state]
        
        A = -torch.exp(self.A_log.float())  # [d_state]
        
        # 简化的状态空间计算
        y = self.selective_scan(x, dt, A, B_proj)
        
        y = y + x * self.D
        y = y * self.activation(res)
        
        return self.out_proj(y)
    
    def selective_scan(self, x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """简化的选择性扫描实现"""
        B, L, D = x.shape
        N = A.shape[0]
        
        # 离散化
        dt = F.softplus(dt)  # [B, L, N]
        A_discrete = torch.exp(A[None, None, :] * dt)  # [B, L, N]
        B_discrete = dt * B  # [B, L, N]
        
        # 状态传播（简化版）
        states = torch.zeros(B, N, device=x.device, dtype=x.dtype)
        outputs = []
        
        for i in range(L):
            states = states * A_discrete[:, i] + B_discrete[:, i] * x[:, i:i+1, :D//N].mean(dim=-1)
            output = states
            outputs.append(output)
        
        y = torch.stack(outputs, dim=1)  # [B, L, N]
        y = repeat(y, 'b l n -> b l (n d)', d=D//N)
        
        return y

class VisionMambaEncoder(nn.Module):
    """
    Vision Mamba编码器，结合Mamba和KAN技术
    """
    def __init__(self, 
                 dim: int = 768,
                 num_layers: int = 12,
                 image_size: int = 256,
                 patch_size: int = 16,
                 num_classes: int = 1000,
                 d_state: int = 16):
        super().__init__()
        
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim))
        
        # Mamba层
        self.layers = nn.ModuleList([
            MambaBlock(dim, d_state) for _ in range(num_layers)
        ])
        
        # KAN层用于特征变换
        self.kan_layers = nn.ModuleList([
            KANLinear(dim, dim) for _ in range(num_layers // 3)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, dim, H/16, W/16]
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = x + self.pos_embed
        
        # Mamba layers with KAN enhancement
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # 每3层使用一次KAN增强
            if i % 3 == 2 and i // 3 < len(self.kan_layers):
                kan_layer = self.kan_layers[i // 3]
                # 对序列进行KAN变换
                B, L, D = x.shape
                x_flat = x.view(-1, D)
                x_enhanced = kan_layer(x_flat)
                x = x_enhanced.view(B, L, D) + x  # 残差连接
        
        x = self.norm(x)
        return x

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
    结合Vision Mamba和KAN的改进FSRA模型
    """
    def __init__(self, 
                 image_size: int = 256,
                 patch_size: int = 16,
                 dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 num_classes: int = 701,  # University-1652类别数
                 d_state: int = 16):
        super().__init__()
        
        # Vision Mamba编码器
        self.encoder = VisionMambaEncoder(
            dim=dim,
            num_layers=num_layers,
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            d_state=d_state
        )
        
        # 跨视角对齐模块
        self.cross_view_align = CrossViewAlignmentModule(dim, num_heads)
        
        # 特征融合
        self.fusion_kan = KANLinear(dim * 2, dim)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, num_classes)
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, satellite_img: torch.Tensor, drone_img: Optional[torch.Tensor] = None):
        """前向传播
        兼容两种调用方式：
        1. 单输入 (用于分类/特征抽取)：model(img) -> {'logits': logits, 'features': feat}
        2. 双输入 (跨视角对齐)：model(sat_img, drone_img) -> tuple 同原设计
        """
        # 如果只提供一个输入张量，则执行单视角特征提取/分类逻辑
        if drone_img is None:
            # 编码特征
            feat = self.encoder(satellite_img)           # [B, num_patches, dim]
            global_feat = self.global_pool(feat.transpose(1, 2)).squeeze(-1)  # [B, dim]
            logits = self.classifier(global_feat)
            return {
                'logits': logits,       # 分类 logits [B, num_classes]
                'features': global_feat # 全局特征 [B, dim]
            }

        # ---- 双输入跨视角对齐 ----
        sat_feat = self.encoder(satellite_img)  # [B, num_patches, dim]
        drone_feat = self.encoder(drone_img)    # [B, num_patches, dim]

        # 跨视角对齐
        sat_aligned = self.cross_view_align(sat_feat, drone_feat, drone_feat)
        drone_aligned = self.cross_view_align(drone_feat, sat_feat, sat_feat)

        # 特征融合
        sat_fused = self.fusion_kan(torch.cat([sat_feat, sat_aligned], dim=-1))
        drone_fused = self.fusion_kan(torch.cat([drone_feat, drone_aligned], dim=-1))

        # 全局特征
        sat_global = self.global_pool(sat_fused.transpose(1, 2)).squeeze(-1)
        drone_global = self.global_pool(drone_fused.transpose(1, 2)).squeeze(-1)

        # 分类
        sat_logits = self.classifier(sat_global)
        drone_logits = self.classifier(drone_global)

        return (
            sat_logits, drone_logits, sat_global, drone_global
        )

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