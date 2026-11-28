import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.local = nn.Linear(3, 6)
        nn.init.zeros_(self.local.weight)
        nn.init.zeros_(self.local.bias)

    def _readout(self, obs, h):
        q = obs.book_shares.to(h.dtype).clamp_min(0)
        p = obs.book_ticks.to(h.dtype)
        depth = q.log1p()
        distance = (p - p[..., :1]).abs().clamp_max(10000).log1p()
        local = torch.stack((depth, q / (1 + q), distance), dim=-1)
        scale, offset = self.local(local).tanh().chunk(2, dim=-1)
        # Modulate the head input without feeding the correction back into memory.
        z = h.unflatten(-1, (2, 10, 3))
        return (z + 0.5 * (scale * z + offset)).flatten(-3)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        h, _ = self.gru(x)
        return self._readout(obs, h)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return self._readout(obs, h[0]), h[0]
