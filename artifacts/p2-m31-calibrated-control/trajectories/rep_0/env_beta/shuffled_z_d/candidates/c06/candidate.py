import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.local = nn.Sequential(nn.Linear(3, 8), nn.Tanh())
        self.read = nn.Linear(80, 16)
        self.gru = nn.GRU(32, 60, batch_first=True)

    def _encode(self, obs):
        base = self.encoder(self.raw_features(obs))
        shares = obs.book_shares.to(base.dtype).clamp_min(0)
        ticks = obs.book_ticks.to(torch.int64)
        occupied = shares > 0
        level = torch.arange(10, device=ticks.device, dtype=torch.int64)
        last = torch.where(occupied, level, -1).cummax(dim=-1).values
        prev = torch.cat((torch.full_like(last[..., :1], -1), last[..., :-1]), dim=-1)
        prev_ticks = ticks.gather(-1, prev.clamp_min(0))
        gap = torch.where(occupied & (prev >= 0), (ticks - prev_ticks).abs(), 0)
        offset = torch.where(occupied, (ticks - ticks[..., :1]).abs(), 0)
        local = self.local(torch.stack(
            (shares.log1p(), gap.to(base.dtype).log1p(),
             offset.to(base.dtype).log1p()), dim=-1))
        local = local * occupied.unsqueeze(-1)
        bid, ask = self.read(local.flatten(-2)).tanh().unbind(dim=-2)
        paired = torch.cat(((bid + ask) * 0.5, (bid - ask) * 0.5), dim=-1)
        return (base + 0.25 * paired).tanh()

    def sequence_context(self, obs):
        return self.gru(self._encode(obs))[0]

    def step_context(self, obs, state):
        x = self._encode(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return h[0], h[0]
