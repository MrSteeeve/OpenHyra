import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.flow = nn.Linear(4, 2)
        self.mix = nn.Linear(6, 32, bias=False)
        self.gru = nn.GRU(32, 54, batch_first=True)
        self.cross = nn.Parameter(torch.full((3, 2), -2.0))
        self.register_buffer("retention", torch.tensor([0.60, 0.88, 0.98]))

    def _inputs(self, obs):
        x = self.encoder(self.raw_features(obs))
        e = obs.previous_event.to(dtype=x.dtype)
        return x, self.flow(torch.asinh(e)).tanh()

    def _trace(self, u, prior):
        r = self.retention.to(dtype=u.dtype).reshape(1, 3, 1)
        a = self.cross.sigmoid().to(dtype=u.dtype).unsqueeze(0)
        # Centered intensity coordinates; zero state denotes the baseline.
        # The nonnegative transition has row sums strictly below one.
        z = r * prior + (1 - r) * (
            a * prior.flip((-1,)) + (1 - a) * u.unsqueeze(1)
        )
        return z.flatten(1)

    def sequence_context(self, obs):
        x, u = self._inputs(obs)
        if x.size(1) == 0:
            return x.new_zeros(x.size(0), 0, 60)
        trace = x.new_zeros(x.size(0), 3, 2)
        history = []
        for current in u.unbind(dim=1):
            flat = self._trace(current, trace)
            history.append(flat)
            trace = flat.reshape(x.size(0), 3, 2)
        z = torch.stack(history, dim=1)
        h, _ = self.gru((x + self.mix(z)).tanh())
        return torch.cat((h, z), dim=-1)

    def step_context(self, obs, state):
        x, u = self._inputs(obs)
        if state is None:
            state = x.new_zeros(x.size(0), 60)
        z = self._trace(u, state[:, 54:].reshape(x.size(0), 3, 2))
        y = (x + self.mix(z)).tanh()
        _, h = self.gru(y.unsqueeze(1), state[:, :54].unsqueeze(0).contiguous())
        context = torch.cat((h[0], z), dim=-1)
        return context, context
