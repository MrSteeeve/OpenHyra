import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.local = nn.Linear(4, 12)
        self.readout = nn.Linear(24, 60, bias=False)
        self.gain = nn.Parameter(torch.tensor(0.1))

    def _context(self, obs, h):
        q = obs.book_shares.to(h.dtype).clamp_min(0).log1p()
        p = obs.book_ticks
        gap = (p[..., :1] - p).abs().to(h.dtype).log1p()
        rank = torch.arange(10, device=h.device, dtype=h.dtype)
        rank = (rank / 9).expand_as(q)
        thin = (1 + q).reciprocal()
        x = torch.stack((q, gap, rank, thin), dim=-1)
        w = (thin - gap - rank).softmax(dim=-1)
        local = (self.local(x).tanh() * w.unsqueeze(-1)).sum(dim=-2)
        return h + self.gain.tanh() * self.readout(local.flatten(-2)).tanh()

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        h, _ = self.gru(x)
        return self._context(obs, h)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return self._context(obs, h[0]), h[0]
