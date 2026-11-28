import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.side = nn.Sequential(nn.Linear(20, 24), nn.Tanh(), nn.Linear(24, 16))
        self.curve = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 16))
        self.gru = nn.GRU(32, 60, batch_first=True)

    def _encode(self, obs):
        base = self.encoder(self.raw_features(obs))
        shares = obs.book_shares.to(base.dtype).clamp_min(0)
        ticks = obs.book_ticks.to(torch.int64)
        live = shares > 0
        distance = (ticks - ticks[..., :1]).abs().to(base.dtype)
        distance = torch.where(live, distance, torch.zeros_like(distance))
        profile = self.side(torch.cat((shares.log1p(), distance.log1p()), dim=-1))

        # Each queue spans an interval in cumulative removal size.
        end = shares.cumsum(dim=-1)
        start = (end - shares).clamp_min(0)
        knots = torch.stack(
            (start.log1p(), end.log1p(), distance.log1p(), shares.log1p()), dim=-1
        )
        values = self.curve(knots) * live.unsqueeze(-1)
        pooled = values.sum(dim=-2) / live.sum(dim=-1, keepdim=True).clamp_min(1)
        bid, ask = (profile + pooled).tanh().unbind(dim=-2)
        paired = torch.cat(((bid + ask) * 0.5, (bid - ask) * 0.5), dim=-1)
        return (base + 0.25 * paired).tanh()

    def sequence_context(self, obs):
        return self.gru(self._encode(obs))[0]

    def step_context(self, obs, state):
        x = self._encode(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return h[0], h[0]
