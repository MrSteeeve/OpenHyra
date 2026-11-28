import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.drive = nn.Linear(7, 2)
        self.gru = nn.GRU(38, 54, batch_first=True)

    def _inputs(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        q = obs.book_shares.to(x.dtype).clamp_min(0).log1p()
        near = q[..., :3].mean(dim=-1)
        deep = q[..., 3:].mean(dim=-1)
        p = (near[..., 0] - near[..., 1]).unsqueeze(-1)
        d = (deep[..., 0] - deep[..., 1]).unsqueeze(-1)
        u = torch.cat(
            (obs.previous_event.to(x.dtype), obs.is_own.to(x.dtype).unsqueeze(-1),
             p, d), dim=-1
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
