import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.queue = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 9))
        self.mix = nn.Linear(16, 32, bias=False)
        self.gru = nn.GRU(32, 60, batch_first=True)

    def encode(self, obs):
        base = self.encoder(self.raw_features(obs))
        depth = obs.book_shares.to(base.dtype).clamp_min(0)
        depth = depth.log1p()
        depth = depth / (1 + depth)
        ticks = obs.book_ticks.to(base.dtype)
        gap = (ticks - ticks[..., :1]).abs()
        gap = gap / (1 + gap)
        rank = torch.arange(10, device=base.device, dtype=torch.int64)
        rank = (rank.to(base.dtype) / 9).expand_as(depth)
        own = obs.is_own.to(base.dtype)[..., None, None].expand_as(depth)
        local = self.queue(torch.stack((depth, gap, rank, own), dim=-1))
        gate = local[..., :1].sigmoid()
        pooled = (gate * local[..., 1:].tanh()).mean(dim=-2)
        queue_context = self.mix(pooled.flatten(-2))
        return (base + queue_context).tanh()

    def sequence_context(self, obs):
        return self.gru(self.encode(obs))[0]

    def step_context(self, obs, state):
        x = self.encode(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return h[0], h[0]
