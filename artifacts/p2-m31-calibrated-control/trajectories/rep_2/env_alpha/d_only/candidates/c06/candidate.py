import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.queue = nn.Sequential(nn.Linear(3, 12), nn.Tanh(), nn.Linear(12, 6))
        nn.init.zeros_(self.queue[-1].weight)
        nn.init.zeros_(self.queue[-1].bias)

    def _readout(self, obs, h):
        q = obs.book_shares.to(h.dtype).clamp_min(0)
        p = obs.book_ticks.to(h.dtype)
        distance = (p - p[..., :1]).abs()
        local = torch.stack((q.log1p(), 1 / (1 + q), distance.log1p()), dim=-1)
        gate, shift = self.queue(local).chunk(2, dim=-1)
        slots = h.unflatten(-1, (2, 10, 3))
        slots = slots * (1 + 0.5 * gate.tanh()) + 0.5 * shift.tanh()
        return slots.flatten(-3)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        h, _ = self.gru(x)
        return self._readout(obs, h)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        _, h = self.gru(x.unsqueeze(1), state.unsqueeze(0))
        return self._readout(obs, h[0]), h[0]
