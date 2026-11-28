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

    def context(self, obs, h):
        q = obs.book_shares.to(h.dtype).clamp_min(0)
        live = q > 0
        best = live.to(torch.int64).argmax(dim=-1, keepdim=True)
        ticks = obs.book_ticks
        d = (ticks - ticks.gather(-1, best)).abs().to(h.dtype)
        d = torch.where(live, d, 0)
        depth = q.log1p()
        share = q / (1 + q.sum(dim=-1, keepdim=True))
        u = self.local(torch.stack((depth, share, d / (1 + d)), -1)).tanh()
        slots = h.reshape(*h.shape[:-1], 2, 10, 3)

        # Align both ladders in mirrored price coordinates.
        def kernel(other_d, other_depth, other_live):
            gap = (d.unsqueeze(-1) - other_d.unsqueeze(-2)).abs()
            score = other_depth.unsqueeze(-2) - gap.log1p() * 2
            return score.masked_fill(~other_live.unsqueeze(-2), -1e4)

        axis = -2
        opposite = kernel(d.flip((axis,)), depth.flip((axis,)),
                          live.flip((axis,)))
        v = opposite.softmax(-1) @ u.flip((-3,))
        v = v * live.flip((axis,)).any(-1, keepdim=True).unsqueeze(-1)
        paired = torch.cat((u * v, u - v), -1)
        gain, shift = (self.readout(u) + self.pair(paired)).split(3, -1)
        keys = torch.cat((u, u.flip((-3,))), -2)
        values = torch.cat((slots, slots.flip((-3,))), -2)
        prior = torch.cat((kernel(d, depth, live), opposite - 1), -1)
        weight = (u @ keys.transpose(-1, -2) / 4 + prior).softmax(-1)
        has_depth = live.any(dim=-1).any(dim=-1)[..., None, None, None]
        delta = (weight @ values - slots) * has_depth
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
