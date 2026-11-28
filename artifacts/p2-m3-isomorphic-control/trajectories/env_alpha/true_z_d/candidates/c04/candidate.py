import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.depth_gate = nn.Linear(10, 60)
        self.local = nn.Linear(32, 60)
        nn.init.zeros_(self.depth_gate.weight)
        nn.init.zeros_(self.depth_gate.bias)
        nn.init.zeros_(self.local.weight)
        nn.init.zeros_(self.local.bias)

    def encode(self, obs):
        return self.encoder(self.raw_features(obs)).tanh()

    def readout(self, obs, x, h):
        depth = obs.book_shares.reshape(*x.shape[:-1], 10).to(x)
        depth = depth.clamp(0, 1e8).log1p()
        depth = depth / (1 + depth)
        gate = self.depth_gate(depth).tanh()
        return h + gate * (h - x.mean(dim=-1, keepdim=True)) + self.local(x)

    def sequence_context(self, obs):
        x = self.encode(obs)
        h, _ = self.gru(x)
        return self.readout(obs, x, h)

    def step_context(self, obs, state):
        x = self.encode(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        h = h[0]
        return self.readout(obs, x, h), h
