import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.local = nn.Linear(3, 4)
        self.mix = nn.Linear(80, 32, bias=False)
        self.readout = nn.Linear(24, 32, bias=False)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.readout.weight)

    def _encode(self, obs):
        x = self.encoder(self.raw_features(obs))
        depth = obs.book_shares.to(dtype=x.dtype).clamp_min(0).log1p()
        neighbor = torch.cat((depth[..., 1:], depth[..., -1:]), dim=-1)
        own = obs.is_own.to(dtype=x.dtype)[..., None, None].expand_as(depth)
        q = torch.stack(
            (depth / (1 + depth), (depth - neighbor).tanh(), own),
            dim=-1,
        )
        local = self.local(q).tanh()
        gain = 1 + 0.5 * self.mix(local.flatten(start_dim=-3)).tanh()

        # Pool each side separately at three distances from its touch.
        ticks = obs.book_ticks.to(dtype=torch.float64)
        gap = (ticks - ticks[..., :1]).abs().clamp_max(128).to(x.dtype)
        scales = x.new_tensor([1, 4, 16])
        weight = torch.exp(-gap.unsqueeze(-1) / scales)
        thin = (1 + depth).reciprocal()
        pooled = torch.einsum("...slk,...slc->...skc",
                              weight * thin.unsqueeze(-1), local)
        pooled = pooled / weight.sum(dim=-2).clamp_min(1).unsqueeze(-1)
        delta = self.readout(pooled.flatten(start_dim=-3))
        return (x * gain + 0.5 * delta.tanh()).tanh()

    def sequence_context(self, obs):
        return self.gru(self._encode(obs))[0]

    def step_context(self, obs, state):
        x = self._encode(obs)
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        _, h = self.gru(x.unsqueeze(1), state.unsqueeze(0))
        return h[0], h[0]
