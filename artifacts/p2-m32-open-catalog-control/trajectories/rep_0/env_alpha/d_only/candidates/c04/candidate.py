import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.thin_bias = nn.Parameter(torch.zeros(2, 10))
        self.queue_readout = nn.Linear(40, 60, bias=False)
        self.queue_key = nn.Linear(20, 12, bias=False)
        self.history_key = nn.Linear(60, 12, bias=False)
        self.interaction_out = nn.Linear(12, 60, bias=False)

    def _readout(self, obs, h):
        q = obs.book_shares.to(h.dtype).clamp_min(0).log1p()
        thin = (self.thin_bias - q).sigmoid()
        ticks = obs.book_ticks.to(torch.int64)
        offset = (ticks - ticks[..., :1]).to(h.dtype)
        flat = thin.flatten(-2)
        local = torch.cat(
            (flat, (thin * (offset / 10).tanh()).flatten(-2)), dim=-1
        )
        joint = self.queue_key(flat).tanh() * self.history_key(h).tanh()
        return h + (self.queue_readout(local) + self.interaction_out(joint)).tanh()

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        h, _ = self.gru(x)
        return self._readout(obs, h)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return self._readout(obs, h[0]), h[0]
