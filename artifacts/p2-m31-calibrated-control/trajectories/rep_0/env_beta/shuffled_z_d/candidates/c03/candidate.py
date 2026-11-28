import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.level = nn.Sequential(nn.Linear(5, 24), nn.Tanh())
        self.score = nn.Linear(24, 1)
        self.value = nn.Linear(24, 16)
        self.side = nn.Linear(32, 16)
        self.gru = nn.GRU(32, 60, batch_first=True)

    def _encode(self, obs):
        base = self.encoder(self.raw_features(obs))
        shares = obs.book_shares.to(base.dtype).clamp_min(0)
        ticks = obs.book_ticks.to(torch.int64)
        occupied = shares > 0
        distance = (ticks - ticks[..., :1]).abs().to(base.dtype)
        distance = torch.where(occupied, distance, 0.0)
        depth = shares.log1p()
        touch = depth[..., :1].expand_as(depth)
        rank = torch.arange(10, device=depth.device, dtype=torch.int64)
        rank = (rank.to(base.dtype) / 9).expand_as(depth)
        levels = self.level(torch.stack(
            (depth, distance.log1p(), touch, depth - touch, rank), dim=-1
        ))
        valid = occupied & (rank > 0)
        scores = self.score(levels).squeeze(-1)
        weights = scores.masked_fill(~valid, -1e4).softmax(dim=-1)
        values = self.value(levels).tanh()
        behind = (values * (weights * valid).unsqueeze(-1)).sum(dim=-2)
        sides = self.side(torch.cat((values[..., 0, :], behind), dim=-1))
        bid, ask = sides.tanh().unbind(dim=-2)
        paired = torch.cat(((bid + ask) * 0.5, (bid - ask) * 0.5), dim=-1)
        return (base + 0.25 * paired).tanh()

    def sequence_context(self, obs):
        return self.gru(self._encode(obs))[0]

    def step_context(self, obs, state):
        x = self._encode(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return h[0], h[0]
