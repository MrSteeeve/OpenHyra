import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.side = nn.Sequential(nn.Linear(20, 24), nn.Tanh(), nn.Linear(24, 16))
        self.gate = nn.Linear(16, 16)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.gru = nn.GRU(32, 60, batch_first=True)

    def _encode(self, obs):
        base = self.encoder(self.raw_features(obs))
        shares = obs.book_shares.to(base.dtype).clamp_min(0)
        ticks = obs.book_ticks.to(torch.int64)
        distance = (ticks - ticks[..., :1]).abs().to(base.dtype)
        distance = torch.where(shares > 0, distance, torch.zeros_like(distance))
        profiles = torch.cat((shares.log1p(), distance.log1p()), dim=-1)
        bid, ask = self.side(profiles).tanh().unbind(dim=-2)
        even = (bid + ask) * 0.5
        odd = (bid - ask) * 0.5
        # Exchange-invariant gain preserves the odd branch's reflection parity.
        odd = odd * (1 + 0.5 * self.gate(even).tanh())
        return (base + 0.25 * torch.cat((even, odd), dim=-1)).tanh()

    def sequence_context(self, obs):
        return self.gru(self._encode(obs))[0]

    def step_context(self, obs, state):
        x = self._encode(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return h[0], h[0]
