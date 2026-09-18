from __future__ import annotations
import math
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

CONCEPT_SPECS = {
    "subtlety": ("ordinal", 5),
    "internalStructure": ("categorical", 4),
    "calcification": ("categorical", 6),
    "sphericity": ("ordinal", 5),
    "margin": ("ordinal", 5),
    "lobulation": ("ordinal", 5),
    "spiculation": ("ordinal", 5),
    "texture": ("ordinal", 5),
}

def _groups(ch: int) -> int:
    for g in (8, 6, 4, 3, 2, 1):
        if ch % g == 0:
            return g
    return 1

class ResBlock3D(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, drop: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(_groups(cout), cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(_groups(cout), cout)
        self.drop = nn.Dropout3d(drop) if drop > 0 else nn.Identity()
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            nn.Conv3d(cin, cout, 1, stride=stride, bias=False),
            nn.GroupNorm(_groups(cout), cout),
        )

    def forward(self, x):
        identity = self.skip(x)
        x = F.silu(self.gn1(self.conv1(x)), inplace=True)
        x = self.drop(x)
        x = self.gn2(self.conv2(x))
        return F.silu(x + identity, inplace=True)

def _norm3d(kind: str, channels: int):
    if kind == "batch":
        return nn.BatchNorm3d(channels)
    if kind == "group":
        return nn.GroupNorm(_groups(channels), channels)
    if kind == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    raise ValueError(f"Unsupported encoder normalization: {kind}")


class UNetConvBlock3D(nn.Module):
    """3D U-Net encoder block: two Conv-Norm-ReLU operations."""
    def __init__(self, cin: int, cout: int, norm: str = "batch"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            _norm3d(norm, cout),
            nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False),
            _norm3d(norm, cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SharedUNetEncoder(nn.Module):
    """Shared encoder-only 3D U-Net path with 16->32->64->128->256 channels."""
    def __init__(self, base: int = 16, out_dim: int = 256, drop: float = 0.05,
                 norm: str = "batch", final_channels: int = 256):
        super().__init__()
        if final_channels not in (192, 256):
            raise ValueError("final_channels must be 192 or 256")
        self.stem = UNetConvBlock3D(1, base, norm)
        self.pool1 = nn.MaxPool3d(2)
        self.s1 = UNetConvBlock3D(base, base * 2, norm)
        self.pool2 = nn.MaxPool3d(2)
        self.s2 = UNetConvBlock3D(base * 2, base * 4, norm)
        self.pool3 = nn.MaxPool3d(2)
        self.s3 = UNetConvBlock3D(base * 4, base * 8, norm)
        self.pool4 = nn.MaxPool3d(2)
        self.s4 = UNetConvBlock3D(base * 8, final_channels, norm)
        self.drop = nn.Dropout3d(drop) if drop > 0 else nn.Identity()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Sequential(
            nn.Linear(final_channels, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.s1(self.pool1(x))
        x = self.s2(self.pool2(x))
        x = self.s3(self.pool3(x))
        x = self.s4(self.pool4(x))
        return self.proj(self.pool(self.drop(x)).flatten(1))

class PatchEmbed3D(nn.Module):
    def __init__(self, embed_dim: int = 128, patch: Tuple[int,int,int] = (8,8,8)):
        super().__init__()
        self.proj = nn.Conv3d(1, embed_dim, kernel_size=patch, stride=patch, bias=False)

    def forward(self, x):
        x = self.proj(x)
        grid = x.shape[-3:]
        return x.flatten(2).transpose(1,2), grid

def _sinusoid_1d(length: int, dim: int, device, dtype):
    if dim % 2:
        raise ValueError("sinusoid dimension must be even")
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) *
                    (-math.log(10000.0) / max(dim, 2)))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(dtype=dtype)

def positional_encoding_3d(grid, dim: int, device, dtype):
    # 128 -> 44 + 42 + 42, each axis allocation is even.
    dz = (dim // 3) // 2 * 2
    dy = dz
    dx = dim - dz - dy
    if dx % 2:
        dx -= 1
        dz += 1
    if dz % 2:
        dz -= 1
        dy += 1
    D,H,W = map(int, grid)
    pez = _sinusoid_1d(D, dz, device, dtype)
    pey = _sinusoid_1d(H, dy, device, dtype)
    pex = _sinusoid_1d(W, dx, device, dtype)
    z = pez[:,None,None,:].expand(D,H,W,dz)
    y = pey[None,:,None,:].expand(D,H,W,dy)
    x = pex[None,None,:,:].expand(D,H,W,dx)
    return torch.cat([z,y,x], dim=-1).reshape(1, D*H*W, dim)

class TransformerBlock(nn.Module):
    def __init__(self, dim=128, heads=4, mlp_ratio=2.0, drop=0.15):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))

class ContextTransformer(nn.Module):
    def __init__(self, dim=128, depth=2, heads=4, drop=0.15, patch=(8,8,8)):
        super().__init__()
        self.patch = PatchEmbed3D(dim, patch)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, 2.0, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        t, grid = self.patch(x)
        t = t + positional_encoding_3d(grid, t.shape[-1], t.device, t.dtype)
        for b in self.blocks:
            t = b(t)
        return self.norm(t)

class CrossAttention(nn.Module):
    def __init__(self, q_dim=256, kv_dim=128, out_dim=256, heads=4, drop=0.10):
        super().__init__()
        self.q = nn.Linear(q_dim, out_dim, bias=False)
        self.k = nn.Linear(kv_dim, out_dim, bias=False)
        self.v = nn.Linear(kv_dim, out_dim, bias=False)
        self.attn = nn.MultiheadAttention(out_dim, heads, dropout=drop, batch_first=True)
        self.out = nn.Linear(out_dim, out_dim)

    def forward(self, q, kv):
        a, _ = self.attn(self.q(q), self.k(kv), self.v(kv), need_weights=False)
        return self.out(a)

class FocusedDSHCBN(nn.Module):
    """
    V4.0 focused DS-HCBN:
      local 64^3 + context 96^3
      shared residual/GN CNN branch
      reduced context Transformer
      local-anchored cross-attention with small residual gates
      native mixed concept heads
      latent-primary malignancy + controlled concept residual

    The changes intentionally target the observed train->validation gap rather than
    maximizing training fit.
    """
    def __init__(
        self,
        cnn_base=16,
        fusion_dim=256,
        transformer_dim=128,
        transformer_depth=2,
        transformer_heads=4,
        patch=(8,8,8),
        q_tokens=2,
        dropout=0.25,
        encoder_drop=0.05,
        encoder_norm="batch",
        encoder_final_channels=256,
        context_gate_init=-2.1972246,   # sigmoid ~= 0.10
        attention_gate_init=-2.1972246, # sigmoid ~= 0.10
        concept_scale_init=-0.6190392,  # sigmoid ~= 0.35
    ):
        super().__init__()
        self.encoder = SharedUNetEncoder(
            cnn_base, fusion_dim, encoder_drop, encoder_norm, encoder_final_channels
        )
        self.context_cnn_gate_logit = nn.Parameter(torch.tensor(float(context_gate_init)))
        self.context_transformer = ContextTransformer(
            transformer_dim, transformer_depth, transformer_heads, drop=0.15, patch=patch
        )
        self.q_tokens = q_tokens
        self.q_mlp = nn.Sequential(
            nn.Linear(fusion_dim, q_tokens * fusion_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.cross = CrossAttention(fusion_dim, transformer_dim, fusion_dim, heads=4, drop=0.10)
        self.attn_gate_logit = nn.Parameter(torch.tensor(float(attention_gate_init)))
        self.fuse_norm = nn.LayerNorm(fusion_dim)
        self.fused_drop = nn.Dropout(dropout)

        self.concept_heads = nn.ModuleDict()
        repr_dim = 0
        for name, (kind, ncls) in CONCEPT_SPECS.items():
            out_dim = ncls - 1 if kind == "ordinal" else ncls
            self.concept_heads[name] = nn.Sequential(
                nn.Linear(fusion_dim, 96),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(96, out_dim),
            )
            repr_dim += 1 if kind == "ordinal" else ncls

        self.latent_head = nn.Sequential(
            nn.Linear(fusion_dim, 96),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )
        self.concept_head = nn.Sequential(
            nn.Linear(repr_dim, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.concept_scale_logit = nn.Parameter(torch.tensor(float(concept_scale_init)))

    def concept_representation(self, logits: Dict[str, torch.Tensor]):
        reps = []
        preds = {}
        for name, (kind, ncls) in CONCEPT_SPECS.items():
            z = logits[name]
            if kind == "ordinal":
                p = torch.sigmoid(z)
                expected = 1.0 + p.sum(dim=1, keepdim=True)
                reps.append((expected - 1.0) / float(ncls - 1))
                preds[name] = expected
            else:
                p = torch.softmax(z, dim=1)
                reps.append(p)
                classes = torch.arange(1, ncls + 1, device=z.device, dtype=z.dtype)[None,:]
                preds[name] = (p * classes).sum(dim=1, keepdim=True)
        return torch.cat(reps, dim=1), preds

    def forward(self, x_local, x_context):
        f_local = self.encoder(x_local)
        f_context = self.encoder(x_context)
        g_ctx = torch.sigmoid(self.context_cnn_gate_logit)
        base = self.fuse_norm(f_local + g_ctx * f_context)

        tok_ctx = self.context_transformer(x_context)
        q = self.q_mlp(base).view(base.size(0), self.q_tokens, -1)
        att = self.cross(q, tok_ctx).mean(dim=1)
        g_att = torch.sigmoid(self.attn_gate_logit)
        fused = self.fuse_norm(base + g_att * att)
        fused = self.fused_drop(fused)

        concept_logits = {k: h(fused) for k,h in self.concept_heads.items()}
        concept_repr, concept_predictions = self.concept_representation(concept_logits)

        z_lat = self.latent_head(fused).squeeze(1)
        z_con = self.concept_head(concept_repr).squeeze(1)
        cscale = torch.sigmoid(self.concept_scale_logit)
        z = z_lat + cscale * z_con

        return {
            "mal_logit": z,
            "latent_logit": z_lat,
            "concept_logit": z_con,
            "concept_logits": concept_logits,
            "concept_predictions": concept_predictions,
            "context_cnn_gate": g_ctx.expand(z.shape[0]),
            "attention_gate": g_att.expand(z.shape[0]),
            "concept_scale": cscale.expand(z.shape[0]),
            "fused": fused,
        }
