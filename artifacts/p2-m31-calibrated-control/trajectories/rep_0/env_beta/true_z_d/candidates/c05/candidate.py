import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.local = nn.Linear(3, 12)
        self.readout = nn.Linear(12, 6)
        self.pair = nn.Linear(24, 6, bias=False)
        self.mix = nn.Linear(24, 3, bias=False)
        for layer in (self.readout, self.pair, self.mix):
            nn.init.zeros_(layer.weight)
        nn.init.zeros_(self.readout.bias)
        level = torch.arange(10, dtype=torch.int64)
        i, j = level[:, None], level[None, :]
        self.register_buffer("gap", torch.cat(((i - j).abs(), i + j + 1), -1))
        self.register_buffer("mask", self.gap > 2)

    def context(self, obs, h):
        q = obs.book_shares.to(h.dtype).clamp_min(0)
        share = q / (1 + q.sum(dim=-1, keepdim=True))
        ticks = obs.book_ticks
        distance = (ticks - ticks[..., :1]).to(h.dtype).abs()
        u = self.local(torch.stack(
            (q.log1p(), share, distance / (1 + distance)), dim=-1
        )).tanh()
        v = u.flip(dims=(-3,))
        paired = torch.cat((u * v, u - v), dim=-1)
        gain, shift = (self.readout(u) + self.pair(paired)).split(3, dim=-1)
        slots = h.reshape(*h.shape[:-1], 2, 10, 3)

        # Each side sees its own ladder first, then the opposite ladder.
        keys = torch.cat((u, v), dim=-2)
        values = torch.cat((slots, slots.flip(dims=(-3,))), dim=-2)
        depth = q.log1p()
        depth = torch.cat((depth, depth.flip(dims=(-2,))), dim=-1)
        score = (u @ keys.transpose(-1, -2)) / 4
        score = score + depth.unsqueeze(-2) - self.gap.to(h.dtype)
        weight = score.masked_fill(self.mask, -torch.inf).softmax(dim=-1)
        delta = weight @ values - slots
        adjusted = slots + 0.25 * (
            gain.tanh() * slots + shift.tanh() + self.mix(paired).tanh() * delta
        )
        return adjusted.reshape_as(h)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        h, _ = self.gru(x)
        return self.context(obs, h)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        h = h[0]
        return self.context(obs, h), h
