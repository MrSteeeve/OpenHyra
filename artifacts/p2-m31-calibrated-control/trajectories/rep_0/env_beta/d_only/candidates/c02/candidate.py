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
        self.register_buffer("retention", torch.tensor([0.60, 0.88, 0.98]))

    def _inputs(self, obs):
        x = self.encoder(self.raw_features(obs))
        e = obs.previous_event.to(dtype=x.dtype)
        u = self.flow(torch.asinh(e)).tanh()
        q = obs.book_shares.to(dtype=x.dtype).clamp_min(0)
        touch = q[..., :, 0].mean(dim=-1)
        depth = q.mean(dim=(-2, -1))
        # Use visible depth as a local scale for the preceding order.
        clock = 0.5 + 2 * torch.sigmoid(
            torch.log1p(e[..., 3].abs()) - torch.log1p(touch + depth)
        )
        return x, u, clock

    def _trace(self, u, prior, clock):
        r = self.retention.to(dtype=u.dtype).view(1, 3, 1)
        decay = r.pow(clock[:, None, None])
        return decay * prior + (1 - decay) * u[:, None, :]

    def sequence_context(self, obs):
        x, u, clock = self._inputs(obs)
        b, t, _ = x.shape
        if t == 0:
            return x.new_zeros(b, 0, 60)
        trace = x.new_zeros(b, 3, 2)
        history = []
        for current, elapsed in zip(u.unbind(1), clock.unbind(1)):
            trace = self._trace(current, trace, elapsed)
            history.append(trace.flatten(1))
        z = torch.stack(history, dim=1)
        h, _ = self.gru((x + self.mix(z)).tanh())
        return torch.cat((h, z), dim=-1)

    def step_context(self, obs, state):
        x, u, clock = self._inputs(obs)
        b = x.size(0)
        if state is None:
            state = x.new_zeros(b, 60)
        prior = state[:, 54:].reshape(b, 3, 2)
        z = self._trace(u, prior, clock).flatten(1)
        y = (x + self.mix(z)).tanh()
        _, h = self.gru(y.unsqueeze(1), state[:, :54].unsqueeze(0).contiguous())
        context = torch.cat((h[0], z), dim=-1)
        return context, context
