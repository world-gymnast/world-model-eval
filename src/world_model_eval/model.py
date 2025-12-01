import torch
from torch import nn
import torch.nn.functional as F
import einops
import math
import functools
from typing import Sequence
import sys
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

class AttentionType(StrEnum):
    SPATIAL = "spatial"
    TEMPORAL = "temporal"


class RotaryType(StrEnum):
    STANDARD = "standard"
    PIXEL = "pixel"


@functools.lru_cache
def rope_nd(
    shape: Sequence[int],
    dim: int = 64,
    base: float = 10_000.0,
    rotary_type: RotaryType = RotaryType.STANDARD,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    D = len(shape)
    assert dim % (2 * D) == 0, (
        f"`dim` must be divisible by 2 × D (got dim={dim}, D={D})"
    )

    dim_per_axis = dim // D
    half = dim_per_axis // 2
    if rotary_type == RotaryType.STANDARD:
        inv_freq = 1.0 / (
            base ** (torch.arange(half, device=device, dtype=dtype) / half)
        )
        coords = [torch.arange(n, device=device, dtype=dtype) for n in shape]
    elif rotary_type == RotaryType.PIXEL:
        inv_freq = (
            torch.linspace(1.0, 256.0 / 2, half, device=device, dtype=dtype) * math.pi
        )
        coords = [
            torch.linspace(-1, +1, steps=n, device=device, dtype=dtype) for n in shape
        ]
    else:
        raise NotImplementedError(f"invalid rotary type: {rotary_type}")

    mesh = torch.meshgrid(*coords, indexing="ij")

    embeddings = []
    for pos in mesh:
        theta = pos.unsqueeze(-1) * inv_freq
        emb_axis = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        embeddings.append(emb_axis)
    return torch.cat(embeddings, dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def rope_mix(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    return x * cos + rotate_half(x) * sin


def apply_rope_nd(
    q: torch.Tensor,
    k: torch.Tensor,
    shape: tuple[int, ...],
    rotary_type: RotaryType,
    *,
    base: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = q.shape[-1]
    rope = rope_nd(
        shape, dim, base, rotary_type=rotary_type, dtype=q.dtype, device=q.device
    )
    rope = rope.view(*shape, len(shape), 2, -1)
    cos, sin = rope.unbind(-2)
    cos = cos.reshape(*shape, -1)
    sin = sin.reshape(*shape, -1)

    k_rot = rope_mix(k, cos, sin)

    # temporal attention with caching
    if len(shape) == 1:
        q_offset = k.shape[2] - q.shape[2]

        if q_offset > 0:
            cos_q = cos[q_offset:, :]
            sin_q = sin[q_offset:, :]
            q_rot = rope_mix(q, cos_q, sin_q)
        else:
            q_rot = rope_mix(q, cos, sin)
    else:
        q_rot = rope_mix(q, cos, sin)

    return q_rot, k_rot


class FinalLayer(nn.Module):
    def __init__(self, dim: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 2, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        _, _, H, W, _ = x.shape
        m = self.adaLN_modulation(c)
        m = einops.repeat(m, "b t d -> b t h w d", h=H, w=W).chunk(2, dim=-1)
        x = self.linear(self.norm(x) * (1 + m[1]) + m[0])
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        is_causal: bool,
        attention_type: AttentionType,
        rotary_type: RotaryType = RotaryType.STANDARD,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim
        self.is_causal = is_causal
        self.attention_type = attention_type
        self.rotary_type = rotary_type
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.k_cache_cond: torch.Tensor | None = None
        self.v_cache_cond: torch.Tensor | None = None
        self.k_cache_null: torch.Tensor | None = None
        self.v_cache_null: torch.Tensor | None = None
        self.cache_start: int | None = None
        self.cache_end: int | None = None

    def clear_cache(self):
        """Clear the KV cache."""
        self.k_cache_cond = None
        self.v_cache_cond = None
        self.k_cache_null = None
        self.v_cache_null = None
        self.cache_start = None
        self.cache_end = None

    def _forward_with_cache(self, x: torch.Tensor, start_frame: int, cache_idx: int, cache_type: str = 'cond') -> torch.Tensor:
        """Forward pass with KV caching for temporal attention."""
        B, T, H, W, D = x.shape

        x = einops.rearrange(x, "b t h w d -> (b h w) t d")

        k_cache = self.k_cache_cond if cache_type == 'cond' else self.k_cache_null
        v_cache = self.v_cache_cond if cache_type == 'cond' else self.v_cache_null

        # x contains only the new frames
        qkv_new = self.qkv_proj(x)
        q_new, k_new, v_new = qkv_new.chunk(3, dim=-1)
        q = einops.rearrange(q_new, "B T (head d) -> B head T d", head=self.num_heads)
        k_new = einops.rearrange(k_new, "B T (head d) -> B head T d", head=self.num_heads)
        v_new = einops.rearrange(v_new, "B T (head d) -> B head T d", head=self.num_heads)

        if k_cache is not None:
            rel_start = start_frame - self.cache_start
            rel_end = cache_idx - self.cache_start

            if rel_start < 0 or rel_end > k_cache.shape[2]:
                raise ValueError(
                    f"Cache bounds error: trying to access cache[{rel_start}:{rel_end}] "
                    f"but cache has length {k_cache.shape[2]} "
                    f"(cache_start={self.cache_start}, cache_end={self.cache_end}, "
                    f"start_frame={start_frame}, cache_idx={cache_idx})"
                )

            k_cached = k_cache[:, :, rel_start:rel_end, :]
            v_cached = v_cache[:, :, rel_start:rel_end, :]
            
            k = torch.cat([k_cached, k_new], dim=2)
            v = torch.cat([v_cached, v_new], dim=2)
        else:
            k = k_new
            v = v_new

        if cache_type == 'cond':
            self.k_cache_cond = k.detach()
            self.v_cache_cond = v.detach()
        else:
            self.k_cache_null = k.detach()
            self.v_cache_null = v.detach()

        self.cache_start = start_frame
        self.cache_end = start_frame + k.shape[2]

        sequence_shape = (k.shape[2],)
        q, k = apply_rope_nd(q, k, sequence_shape, rotary_type=self.rotary_type)
        q = einops.rearrange(q, "B head T d -> B head T d")
        k = einops.rearrange(k, "B head T d -> B head T d")
        v = einops.rearrange(v, "B head T d -> B head T d")

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = einops.rearrange(out, "B head seq d -> B seq (head d)")
        out = self.out_proj(out)

        out = einops.rearrange(out, "(b h w) t d -> b t h w d", h=H, w=W)
        return out

    def forward(self, x: torch.Tensor, cache_idx: int | None = None, start_frame: int | None = None, cache_type: str = 'cond'):
        B, T, H, W, D = x.shape

        use_cache = (
            cache_idx is not None
            and start_frame is not None
            and cache_idx > start_frame
            and self.attention_type == AttentionType.TEMPORAL
        )

        if use_cache:
            return self._forward_with_cache(x, start_frame, cache_idx, cache_type)

        if self.attention_type == AttentionType.SPATIAL:
            x = einops.rearrange(x, "b t h w d -> (b t) h w d")
        elif self.attention_type == AttentionType.TEMPORAL:
            x = einops.rearrange(x, "b t h w d -> (b h w) t d")
        else:
            raise NotImplementedError(f"invalid attention type: {self.attention_type}")
        sequence_shape = x.shape[1:-1]

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = einops.rearrange(q, "B ... (head d) -> B head ... d", head=self.num_heads)
        k = einops.rearrange(k, "B ... (head d) -> B head ... d", head=self.num_heads)
        v = einops.rearrange(v, "B ... (head d) -> B head ... d", head=self.num_heads)

        q, k = apply_rope_nd(q, k, sequence_shape, rotary_type=self.rotary_type)
        # Flatten the sequence dimension
        q = einops.rearrange(q, "B head ... d -> B head (...) d")
        k = einops.rearrange(k, "B head ... d -> B head (...) d")
        v = einops.rearrange(v, "B head ... d -> B head (...) d")

        x = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        x = einops.rearrange(x, "B head seq d -> B seq (head d)")
        x = self.out_proj(x)

        if self.attention_type == AttentionType.SPATIAL:
            x = einops.rearrange(x, "(b t) (h w) d -> b t h w d", t=T, h=H, w=W)
        elif self.attention_type == AttentionType.TEMPORAL:
            x = einops.rearrange(x, "(b h w) t d -> b t h w d", h=H, w=W)
        return x


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_type: AttentionType,
        rotary_type: RotaryType,
        is_causal: bool,
    ) -> None:
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6, bias=True)
        )
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads,
            is_causal=is_causal,
            attention_type=attention_type,
            rotary_type=rotary_type,
        )
        self.ffwd = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, cache_idx: int | None = None, start_frame: int | None = None, cache_type: str = 'cond') -> torch.Tensor:
        _, _, H, W, _ = x.shape
        m = self.adaLN_modulation(c)
        m = einops.repeat(m, "b t d -> b t h w d", h=H, w=W).chunk(6, dim=-1)
        x = x + self.attn(self.norm1(x) * (1 + m[1]) + m[0], cache_idx, start_frame, cache_type) * m[2]
        x = x + self.ffwd(self.norm2(x) * (1 + m[4]) + m[3]) * m[5]
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        rope_config: dict[AttentionType, RotaryType] | None = None,
    ) -> None:
        super().__init__()
        self.s_block = DiTBlock(
            dim,
            num_heads,
            is_causal=False,
            attention_type=AttentionType.SPATIAL,
            rotary_type=rope_config[AttentionType.SPATIAL]
            if rope_config
            else RotaryType.STANDARD,
        )
        self.t_block = DiTBlock(
            dim,
            num_heads,
            is_causal=True,
            attention_type=AttentionType.TEMPORAL,
            rotary_type=rope_config[AttentionType.TEMPORAL]
            if rope_config
            else RotaryType.STANDARD,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, cache_idx: int | None = None, start_frame: int | None = None, cache_type: str = 'cond') -> torch.Tensor:
        x = self.s_block(x, c)
        x = self.t_block(x, c, cache_idx, start_frame, cache_type)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = 2,
        dim: int = 1152,
        num_layers: int = 28,
        num_heads: int = 16,
        action_dim: int = 0,
        max_frames: int = 16,
        rope_config: dict[AttentionType, RotaryType] | None = None,
        action_dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.action_dim = action_dim
        self.action_dropout_prob = action_dropout_prob
        self.x_proj = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        self.timestep_mlp = nn.Sequential(
            nn.Linear(256, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )
        self.action_embedder = nn.Linear(action_dim, dim)
        self.blocks = nn.ModuleList(
            [Block(dim, num_heads, rope_config) for _ in range(num_layers)]
        )
        self.final_layer = FinalLayer(dim, patch_size, in_channels)
        self.max_frames = max_frames
        self.initialize_weights()

    def timestep_embedding(
        self, t: torch.Tensor, dim: int = 256, max_period: int = 10000
    ) -> torch.Tensor:
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.timestep_mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.s_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.s_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.t_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.t_block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x = einops.rearrange(x, "b t h w c -> (b t) c h w")
        x = self.x_proj(x)
        x = einops.rearrange(x, "(b t) d h w -> b t h w d", t=T)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(
            x,
            "b h w (p1 p2 c) -> b (h p1) (w p2) c",
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.in_channels,
        )

    def get_null_cond(self, action: torch.Tensor) -> torch.Tensor:
        null_action = torch.zeros_like(action)
        # NOTE: all-zero action is still conditional (meaning "do not move"), so we
        # need to reserve the last component of the action vector to indicate null.
        null_action[..., -1] = 1
        return null_action

    def get_cond(self, t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B, T = t.shape
        t = einops.rearrange(t, "b t -> (b t)")
        t_freq = self.timestep_embedding(t)
        c = self.timestep_mlp(t_freq)
        c = einops.rearrange(c, "(b t) d -> b t d", t=T)
        if self.training and self.action_dropout_prob > 0:
            should_drop = torch.rand((B, 1, 1), device=action.device) < self.action_dropout_prob
            null_action = self.get_null_cond(action)
            action = torch.where(should_drop, null_action, action)
        c += self.action_embedder(action)
        return c

    def clear_kv_cache(self):
        """Clear KV cache in all temporal attention layers."""
        for block in self.blocks:
            block.t_block.attn.clear_cache()

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, action: torch.Tensor,
        cache_idx: int | None = None, start_frame: int | None = None, cache_type: str = 'cond'
    ) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x = self.patchify(x)
        c = self.get_cond(t, action)

        # Only slice input if cache actually exists
        has_cache = any(
            block.t_block.attn.k_cache_cond is not None if cache_type == 'cond'
            else block.t_block.attn.k_cache_null is not None
            for block in self.blocks
        )

        if cache_idx is not None and start_frame is not None and cache_idx > start_frame and has_cache:
             start_rel = cache_idx - start_frame
             x = x[:, start_rel:, ...]
             c = c[:, start_rel:, ...]
             # Update T for subsequent reshapes
             T = x.shape[1]

        for block in self.blocks:
            x = block(x, c, cache_idx, start_frame, cache_type)
        x = self.final_layer(x, c)
        x = einops.rearrange(x, "b t h w d -> (b t) h w d")
        x = self.unpatchify(x)
        x = einops.rearrange(x, "(b t) h w c -> b t h w c", t=T)
        return x
