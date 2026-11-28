import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.local = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 4))
        self.mix = nn.Linear(40, 32, bias=False)
        nn.init.normal_(self.mix.weight, std=0.01)

    def encode(self, obs):
        raw = self.raw_features(obs)
        q = obs.book_shares.to(dtype=raw.dtype).reshape(*raw.shape[:-1], 10)
        q = q.clamp_min(0)
        z = torch.stack((q.log1p(), (1 + q).reciprocal()), dim=-1)
        local = self.local(z).flatten(-2)
        return (self.encoder(raw) + self.mix(local)).tanh()

    def sequence_context(self, obs):
        return self.gru(self.encode(obs))[0]

    def step_context(self, obs, state):
        x = self.encode(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return h[0], h[0]
