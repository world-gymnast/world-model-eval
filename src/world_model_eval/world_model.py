import torch
import einops
from .model import DiT, AttentionType, RotaryType
from .vae import VAE
from .diffusion import Diffusion


class WorldModel:
    def __init__(self, checkpoint_path: str, use_pixel_rope: bool = False, default_cfg: float = 1.0, use_kv_cache: bool = True, rank: int = 0):
        self.device = "cuda:{}".format(rank)
        self.model = (
            DiT(
                in_channels=16,
                patch_size=2,
                dim=1024,
                num_layers=16,
                num_heads=16,
                action_dim=10,
                max_frames=20,
                rope_config={
                    AttentionType.SPATIAL: RotaryType.PIXEL if use_pixel_rope else RotaryType.STANDARD,
                    AttentionType.TEMPORAL: RotaryType.STANDARD,
                },
            )
            .to(self.device)
            .eval()
        )
        state_dict = torch.load(checkpoint_path, weights_only=True)
        if "ema" in state_dict:
            state_dict = state_dict["ema"]
        self.model.load_state_dict(state_dict, strict=True)
        self.vae = VAE().to(self.device).eval()

        self.diffusion = Diffusion(
            timesteps=1000, sampling_timesteps=10, device=self.device
        ).to(self.device)
        self.chunk_size = 1
        self.actions = None
        self.curr_frame = 0
        self.cfg = default_cfg  # Feel free to override this after __init__
        self.use_kv_cache = use_kv_cache

    def reset(self, x):
        x = einops.repeat(x, "b h w c -> b t h w c", t=1)
        self.batch_size = x.shape[0]
        self.xs = self.vae.encode(x)
        self.actions = torch.zeros((self.batch_size, 1, self.model.action_dim), device=self.device)
        self.curr_frame = 1
        # Clear KV cache on reset
        if self.use_kv_cache:
            self.model.clear_kv_cache()

    @torch.no_grad()
    def generate_chunk(self, action_vec):
        """See Diffusion.generate"""
        if self.use_kv_cache:
            self.model.clear_kv_cache()
        if action_vec.dim() == 2:
            # Batch of actions (batch_size, action_dim) -> (batch_size, 1, action_dim)
            action_vec = action_vec.unsqueeze(1)
        
        action_chunk = torch.zeros(
            (self.batch_size, self.chunk_size, self.model.action_dim), device=self.device
        )
        assert self.actions.shape[1] == self.curr_frame
        self.actions = torch.cat([self.actions, action_chunk], dim=1)
        self.actions[:, self.curr_frame : self.curr_frame + self.chunk_size, :] = action_vec

        scheduling_matrix = self.diffusion.generate_pyramid_scheduling_matrix(
            self.chunk_size
        )
        chunk = torch.randn(
            (self.batch_size, self.chunk_size, *self.xs.shape[-3:]), device=self.device
        )
        self.xs = torch.cat([self.xs, chunk], dim=1)

        # Adjust context length
        start_frame = max(0, self.curr_frame + self.chunk_size - self.model.max_frames)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for m in range(scheduling_matrix.shape[0] - 1):
                t, t_next = scheduling_matrix[m], scheduling_matrix[m + 1]
                t = einops.repeat(t, "t -> b t", b=self.batch_size)
                t_next = einops.repeat(t_next, "t -> b t", b=self.batch_size)
                t = torch.cat(
                    (torch.zeros((self.batch_size, self.curr_frame), dtype=torch.long, device=t.device), t), dim=1
                )
                t_next = torch.cat(
                    (torch.zeros((self.batch_size, self.curr_frame), dtype=torch.long, device=t_next.device), t_next), dim=1
                )

                if self.use_kv_cache:
                    window_clean_mask = (t[:, start_frame:] == 0)
                    if window_clean_mask.any():
                        last_clean_rel_idx = window_clean_mask.nonzero()[-1][1].item()
                        cache_idx = start_frame + last_clean_rel_idx + 1
                    else:
                        cache_idx = start_frame
                else:
                    cache_idx = None

                result = self.diffusion.ddim_sample_step(
                    self.model,
                    self.xs[:, start_frame:],
                    self.actions[:, start_frame : self.curr_frame + self.chunk_size],
                    t[:, start_frame:],
                    t_next[:, start_frame:],
                    cfg=self.cfg,
                    cache_idx=cache_idx,
                    start_frame=start_frame,
                )

                # Check if model actually sliced by comparing result shape
                expected_full_length = self.curr_frame + self.chunk_size - start_frame
                if result.shape[1] < expected_full_length:
                    # Model sliced: result only contains frames from cache_idx onwards
                    self.xs[:, cache_idx:] = result
                else:
                    # Model didn't slice: result contains all frames from start_frame
                    self.xs[:, start_frame:] = result

                latest_clean_idx = (t_next == 0).nonzero()[-1][1]
                if latest_clean_idx >= self.curr_frame:
                    xs = self.vae.decode(
                        self.xs[:, latest_clean_idx : latest_clean_idx + 1]
                    )
                    yield latest_clean_idx, xs
        self.curr_frame += self.chunk_size
