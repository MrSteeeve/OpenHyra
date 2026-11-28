import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.drive = nn.Linear(9, 2)
        self.gru = nn.GRU(38, 54, batch_first=True)

    def _inputs(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        q = obs.book_shares.to(x.dtype).clamp_min(0)
        ticks = obs.book_ticks.to(torch.int64)
        distance = (ticks - ticks[..., :1]).abs().to(x.dtype)
        volume = q.new_tensor((32., 128.))
        end = q.cumsum(dim=-1)
        before = torch.cat((torch.zeros_like(end[..., :1]), end[..., :-1]), dim=-1)
        filled = torch.minimum(
            (volume - before.unsqueeze(-1)).clamp_min(0), q.unsqueeze(-1)
        )
        # Bound the feature and account for volume beyond the visible ladder.
        gap = (distance / (1 + distance)).unsqueeze(-1)
        missing = (1 - end[..., -1:] / volume).clamp_min(0)
        cost = (filled * gap / volume).sum(dim=-2) + missing
        u = torch.cat(
            (obs.previous_event.to(x.dtype),
             obs.is_own.to(x.dtype).unsqueeze(-1), cost.flatten(-2)),
            dim=-1,
        )
        return x, self.drive(u).tanh()

    def _advance(self, m, u):
        r = u.new_tensor((0.65, 0.90, 0.98)).reshape(1, 3, 1)
        return r * m + (1 - r) * u.unsqueeze(1)

    def step_context(self, obs, state):
        x, u = self._inputs(obs)
        b = x.size(0)
        if state is None:
            h = x.new_zeros(b, 54)
            m = x.new_zeros(b, 3, 2)
        else:
            h, m = state
        m = self._advance(m, u)
        z = m.flatten(1)
        _, h = self.gru(torch.cat((x, z), dim=-1).unsqueeze(1), h.unsqueeze(0))
        return torch.cat((h[0], z), dim=-1), (h[0], m)

    def sequence_context(self, obs):
        x, u = self._inputs(obs)
        if x.size(1) == 0:
            return x.new_empty(x.size(0), 0, 60)
        m = x.new_zeros(x.size(0), 3, 2)
        traces = []
        for v in u.unbind(dim=1):
            m = self._advance(m, v)
            traces.append(m.flatten(1))
        z = torch.stack(traces, dim=1)
        h, _ = self.gru(torch.cat((x, z), dim=-1))
        return torch.cat((h, z), dim=-1)
