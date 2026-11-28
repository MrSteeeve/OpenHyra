import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.readout = nn.Linear(40, 60, bias=False)
        nn.init.zeros_(self.readout.weight)

    def _profile(self, obs, x):
        q = obs.book_shares.to(dtype=x.dtype).clamp(0, 1e8)
        depth = q.cumsum(dim=-1)
        thin = (1 + depth).rsqrt()
        ticks = obs.book_ticks.to(torch.int64)
        spread = (ticks[..., 1, :1] - ticks[..., 0, :1]).abs()
        gap = (ticks[..., 1:] - ticks[..., :-1]).abs()
        touch = spread.unsqueeze(-2).expand_as(ticks[..., :1])
        width = torch.cat((touch, gap), dim=-1).to(dtype=x.dtype)
        width = width.clamp(0, 1e6).log1p()
        return torch.cat(
            (thin.flatten(-2), (thin * width).flatten(-2)), dim=-1
        )

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        h, _ = self.gru(x)
        return h + self.readout(self._profile(obs, x))

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, last = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        h = last[0]
        return h + self.readout(self._profile(obs, x)), h
