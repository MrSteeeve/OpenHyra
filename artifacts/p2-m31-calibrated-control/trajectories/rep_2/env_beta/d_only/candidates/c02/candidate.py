import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.drive = nn.Linear(7, 2)
        self.clock_gate = nn.Linear(2, 2)
        nn.init.zeros_(self.clock_gate.weight)
        nn.init.zeros_(self.clock_gate.bias)
        self.gru = nn.GRU(38, 54, batch_first=True)

    def _inputs(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        q = obs.book_shares.to(x.dtype).clamp_min(0).log1p()
        near = q[..., :3].mean(dim=-1)
        deep = q[..., 3:].mean(dim=-1)
        p = near[..., :1] - near[..., 1:]
        d = deep[..., :1] - deep[..., 1:]
        e = obs.previous_event.to(x.dtype)
        own = obs.is_own.to(x.dtype).unsqueeze(-1)
        u = self.drive(torch.cat((e, own, p, d), dim=-1)).tanh()

        # Front-to-back depth contrast sets a bounded clock per trace channel.
        contrast = (deep - near).tanh()
        clock = (0.35 * contrast + 0.30 * self.clock_gate(contrast).tanh()).exp()
        return x, u, clock

    def _advance(self, m, u, clock):
        base = u.new_tensor((0.65, 0.90, 0.98)).view(1, 3, 1)
        weight = -torch.expm1(base.log() * clock.unsqueeze(1))
        return m + weight * (u.unsqueeze(1) - m)

    def step_context(self, obs, state):
        x, u, clock = self._inputs(obs)
        if state is None:
            h = x.new_zeros(x.size(0), 54)
            m = x.new_zeros(x.size(0), 3, 2)
        else:
            h, m = state
        m = self._advance(m, u, clock)
        z = m.flatten(1)
        _, h = self.gru(torch.cat((x, z), dim=-1).unsqueeze(1), h.unsqueeze(0))
        return torch.cat((h[0], z), dim=-1), (h[0], m)

    def sequence_context(self, obs):
        x, u, clock = self._inputs(obs)
        if x.size(1) == 0:
            return x.new_empty(x.size(0), 0, 60)
        m = x.new_zeros(x.size(0), 3, 2)
        traces = []
        for v, c in zip(u.unbind(1), clock.unbind(1)):
            m = self._advance(m, v, c)
            traces.append(m.flatten(1))
        z = torch.stack(traces, dim=1)
        h, _ = self.gru(torch.cat((x, z), dim=-1))
        return torch.cat((h, z), dim=-1)
