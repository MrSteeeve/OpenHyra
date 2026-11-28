import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.queue_gate = nn.Parameter(torch.zeros(2, 10, 3))
        self.queue_readout = nn.Linear(60, 60, bias=False)
        nn.init.zeros_(self.queue_readout.weight)

    def _context(self, hidden, shares):
        depth = shares.to(dtype=hidden.dtype).clamp(0, 1e8)
        touch = depth[..., :1]
        features = torch.stack(
            (
                depth.log1p() / 16.0,
                1.0 / (1.0 + depth),
                (depth - touch) / (1.0 + depth + touch),
            ),
            dim=-1,
        )
        gate = (features + self.queue_gate).tanh()
        local = hidden.reshape(*hidden.shape[:-1], 2, 10, 3)
        return hidden + self.queue_readout((local * gate).flatten(-3))

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        hidden, _ = self.gru(x)
        return self._context(hidden, obs.book_shares)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return self._context(h[0], obs.book_shares), h[0]
