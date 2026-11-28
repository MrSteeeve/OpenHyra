import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.local = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 9))
        self.mix = nn.Linear(32, 32, bias=False)
        self.gru = nn.GRU(32, 60, batch_first=True)

    def encode(self, obs):
        x = self.encoder(self.raw_features(obs))
        q = obs.book_shares.to(x.dtype).clamp(0, 1e9).log1p()
        p = obs.book_ticks.to(x.dtype)
        gap = (p - p[..., :1]).abs().clamp(max=1e6).log1p()
        rank = torch.arange(10, device=x.device, dtype=x.dtype) / 9
        z = self.local(torch.stack((q / 8, gap / 4, rank.expand_as(q)), -1))
        w = (z[..., :1] - 2 * rank[..., None]).softmax(dim=-2)
        v = z[..., 1:].tanh()
        summary = torch.cat((v[..., 0, :], (w * v).sum(dim=-2)), dim=-1)
        return (x + self.mix(summary.flatten(-2))).tanh()

    def sequence_context(self, obs):
        return self.gru(self.encode(obs))[0]

    def step_context(self, obs, state):
        x = self.encode(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return h[0], h[0]
