import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.queue_encoder = nn.Linear(2, 2)
        self.queue_readout = nn.Linear(40, 60, bias=False)
        nn.init.zeros_(self.queue_readout.weight)

    def _readout(self, obs, h):
        shares = obs.book_shares.to(dtype=h.dtype).clamp_min(0)
        scarcity = (1 + shares).reciprocal()
        touch = scarcity[..., :1].expand_as(scarcity)
        local = torch.stack((scarcity, touch), dim=-1)
        repair = scarcity.unsqueeze(-1) * self.queue_encoder(local).tanh()
        return h + self.queue_readout(repair.flatten(start_dim=-3))

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        h, _ = self.gru(x)
        return self._readout(obs, h)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return self._readout(obs, h[0]), h[0]
