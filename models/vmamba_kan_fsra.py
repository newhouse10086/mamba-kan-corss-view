import torch.nn as nn
import torch
from einops import rearrange

# --- Model Version ---
MODEL_VERSION = "2.2"
print(f"[FSRA-VMK Model] Loading version {MODEL_VERSION}")

class MambaBlock(nn.Module):
    """
    简化的Mamba风格块
    """
    def __init__(self, dim: int, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = int(expand * dim)
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[..., :x.shape[-1]]
        x = rearrange(x, "b d l -> b l d")
        y = self.act(x) * self.act(z)
        output = self.dropout(self.out_proj(y))
        return output + residual

class VisionMambaEncoder(nn.Module):
    """
    轻量化的Vision Mamba编码器
    """
    def __init__(self, embed_dim=384, num_layers=12, image_size=256, patch_size=16):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        
        self.layers = nn.ModuleList([
            MambaBlock(dim=embed_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class VMambaKANFSRA(nn.Module):
    """
    轻量化、对齐官方FSRA参数量的VMamba-KAN模型
    """
    def __init__(self, num_classes: int, block: int = 1, backbone: str = 'VIT-S'):
        super().__init__()
        
        if backbone == 'VIT-S':
            model_dim = 384
            model_depth = 12
        else: # VIT-B or default
            model_dim = 768
            model_depth = 12
        
        self.backbone = VisionMambaEncoder(
            embed_dim=model_dim,
            num_layers=model_depth
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(model_dim * block, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.block = block

    def forward(self, x):
        x = self.backbone(x)
        features = []
        for i in range(self.block):
            chunk_size = x.size(1) // self.block
            start, end = i * chunk_size, (i + 1) * chunk_size
            features.append(x[:, start:end, :].mean(dim=1))
        
        feat = torch.cat(features, dim=1)
        logits = self.classifier(feat)
        return logits, feat 