import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.depth = nn.Linear(20, 16)
        self.gru = nn.GRU(32, 60, batch_first=True)

    def _encode(self, obs):
        base = self.encoder(self.raw_features(obs))
        q = obs.book_shares.to(dtype=base.dtype).clamp_min(0)
        depth = torch.cat((q.log1p(), q.cumsum(dim=-1).log1p()), dim=-1)
        sides = self.depth(depth).tanh()
        bid, ask = sides.unbind(dim=-2)
        shared = torch.cat((bid + ask, bid - ask), dim=-1) * 0.5
        own = obs.is_own.to(dtype=base.dtype).unsqueeze(-1)
        return (base + 0.25 * (1.0 + own) * shared).tanh()

    def sequence_context(self, obs):
        return self.gru(self._encode(obs))[0]

    def step_context(self, obs, state):
        x = self._encode(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return h[0], h[0]
