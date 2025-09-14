import torch

class StateDynamicsModel(torch.nn.Module):
    """
    Transformer state dynamics model.

    - Inputs:
        state0: (B, state_dim)
        actions: (B, T*frame_skip, action_dim)

    - Outputs:
        next_states: (B, T, state_dim) predicted states [s1, ..., sT]
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 6,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        max_len: int = 64,
        frame_skip: int = 1,
        ckpt: str | None = None
    ) -> None:
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        self.state_dim = int(state_dim)
        self.combined_action_dim = int(action_dim) * frame_skip
        self.frame_skip = frame_skip
        self.d_model = int(d_model)

        self.state_proj = torch.nn.Linear(self.state_dim, self.d_model)
        self.action_proj = torch.nn.Linear(self.combined_action_dim, self.d_model)
        self.pos_emb = torch.nn.Embedding(max_len + 1, self.d_model)

        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.delta_head = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, self.d_model),
            torch.nn.GELU(),
            torch.nn.Linear(self.d_model, self.state_dim),
        )

        if ckpt is not None:
            state_dict = torch.load(ckpt, weights_only=True)
            self.load_state_dict(state_dict['ema'], strict=True)

    def forward(self, state0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        - Inputs:
            state0: (B, state_dim)
            actions: (B, T*frame_skip, action_dim)

        - Outputs:
            next_states: (B, T, state_dim) predicted states excluding s0
        """
        assert state0.dim() == 2, "state0 must be (B, state_dim)"
        assert actions.dim() == 3, "actions must be (B, T * frame_skip, action_dim)"
        B, T, A = actions.shape
        actions = actions.reshape(B, T // self.frame_skip, A * self.frame_skip)
        assert A == self.combined_action_dim // self.frame_skip
        assert state0.shape[1] == self.state_dim

        state_tok = self.state_proj(state0)  # (B, D)
        action_toks = self.action_proj(actions)  # (B, T, D)
        toks = torch.cat([state_tok.unsqueeze(1), action_toks], dim=1)  # (B, T+1, D)

        pos_ids = torch.arange(0, T + 1, device=toks.device).unsqueeze(0).expand(B, -1)
        toks = toks + self.pos_emb(pos_ids)

        h = self.encoder(toks)  # (B, T+1, D)
        h_actions = h[:, 1:, :]  # (B, T, D)
        deltas = self.delta_head(h_actions)  # (B, T, state_dim)

        next_states = state0.unsqueeze(1) + deltas.cumsum(dim=1)
        return next_states
    

# def main():
#     model = StateDynamicsModel(
#         state_dim=7,
#         action_dim=10,
#         d_model=256,
#         nhead=4,
#         num_layers=12,
#         frame_skip=1,
#         ckpt="checkpoints/bridge_v2_state_based_1frame.pt"
#     ).cuda()

#     random_action = torch.randn((1, 1, 10)).cuda()
#     random_state = torch.randn((1, 7)).cuda()
#     # outputs next state
#     print(model(random_state, random_action).shape)

# if __name__ == "__main__":
#     main()

