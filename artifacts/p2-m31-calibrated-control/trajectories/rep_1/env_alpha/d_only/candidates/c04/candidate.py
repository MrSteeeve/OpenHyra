import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.queue = nn.Linear(3, 8)
        self.readout = nn.Linear(160, 60, bias=False)
        nn.init.zeros_(self.readout.weight)

    def _context(self, obs, h):
        q = obs.book_shares.to(dtype=h.dtype).clamp(0, 1e9)
        prefix = q.cumsum(dim=-1)
        features = torch.stack(
            (q.log1p() / 10, prefix.log1p() / 10,
             q / (1 + prefix)),
            dim=-1,
        )
        local = self.queue(features).tanh().flatten(-3)
        return h + self.readout(local).tanh()

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        h, _ = self.gru(x)
        return self._context(obs, h)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return self._context(obs, h[0]), h[0]
