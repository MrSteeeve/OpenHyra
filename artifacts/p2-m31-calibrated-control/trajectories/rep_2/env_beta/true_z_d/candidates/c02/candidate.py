import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.local = nn.Conv1d(2, 4, 3, padding=1)
        self.gate = nn.Linear(80, 32)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def _encode(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        shares = obs.book_shares.to(dtype=x.dtype)
        depth = shares.clamp(0, 1e8).log1p()
        ticks = obs.book_ticks
        gap = (ticks - ticks[..., :1]).to(dtype=x.dtype)
        distance = gap.abs().clamp_max(1e6).log1p()
        # Apply the same local filter independently to both sides.
        local = torch.stack((depth, distance), dim=-2)
        y = self.local(local.reshape(-1, 2, 10)).tanh()
        y = y.reshape(*x.shape[:-1], 2 * 4 * 10)
        return x * (1.0 + self.gate(y).tanh())

    def sequence_context(self, obs):
        return self.gru(self._encode(obs))[0]

    def step_context(self, obs, state):
        x = self._encode(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return h[0], h[0]
