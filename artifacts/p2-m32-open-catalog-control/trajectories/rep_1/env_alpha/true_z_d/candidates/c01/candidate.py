import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.repair = nn.Linear(20, 32, bias=False)
        nn.init.zeros_(self.repair.weight)

    def _encode(self, obs):
        base = self.encoder(self.raw_features(obs))
        q = obs.book_shares.to(torch.float32).clamp_min(0)
        ticks = obs.book_ticks.to(torch.int64)
        gap = (ticks[..., 1:] - ticks[..., :-1]).abs().to(q.dtype)
        depth = q.log1p()
        scale = 1.0 + depth.mean(dim=(-2, -1), keepdim=True)
        shallow = scale / (scale + depth)
        backup = depth[..., 1:] / (scale + depth[..., 1:])
        adjacent = (
            backup
            * (q[..., 1:] > 0)
            / (1.0 + gap)
        )
        adjacent = torch.cat(
            (adjacent, torch.zeros_like(depth[..., :1])), dim=-1
        )
        fragile = shallow * (1.0 - adjacent)
        return (base + self.repair(fragile.flatten(-2).to(base.dtype))).tanh()

    def sequence_context(self, obs):
        return self.gru(self._encode(obs))[0]

    def step_context(self, obs, state):
        x = self._encode(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return h[0], h[0]
