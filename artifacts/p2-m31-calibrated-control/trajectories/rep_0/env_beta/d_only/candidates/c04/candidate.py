import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.flow = nn.Linear(4, 2)
        self.mix = nn.Linear(6, 32, bias=False)
        self.depth_mix = nn.Linear(20, 32, bias=False)
        nn.init.zeros_(self.depth_mix.weight)
        self.gru = nn.GRU(32, 54, batch_first=True)
        self.register_buffer("retention", torch.tensor([0.60, 0.88, 0.98]))

    def _inputs(self, obs):
        x = self.encoder(self.raw_features(obs))
        e = obs.previous_event.to(dtype=x.dtype)
        u = self.flow(torch.asinh(e)).tanh()
        q = obs.book_shares.to(dtype=x.dtype).clamp(0, 1e7)
        # Depth ahead of each queue, separately on each side.
        ahead = torch.cat(
            (torch.zeros_like(q[..., :1]), q[..., :-1].cumsum(-1)), dim=-1
        )
        scale = 1 + q.mean(dim=(-2, -1), keepdim=True)
        reach = scale / (scale + ahead)
        local = scale / (scale + q)
        interaction = u.unsqueeze(-1) * reach * local
        return x + self.depth_mix(interaction.flatten(-2)), u

    def _trace(self, u, prior):
        r = self.retention.to(dtype=u.dtype).view(1, 3, 1)
        return r * prior + (1 - r) * u.unsqueeze(1)

    def sequence_context(self, obs):
        x, u = self._inputs(obs)
        if x.size(1) == 0:
            return x.new_zeros(x.size(0), 0, 60)
        trace = x.new_zeros(x.size(0), 3, 2)
        history = []
        for current in u.unbind(dim=1):
            trace = self._trace(current, trace)
            history.append(trace.flatten(1))
        z = torch.stack(history, dim=1)
        h, _ = self.gru((x + self.mix(z)).tanh())
        return torch.cat((h, z), dim=-1)

    def step_context(self, obs, state):
        x, u = self._inputs(obs)
        if state is None:
            state = x.new_zeros(x.size(0), 60)
        trace = self._trace(u, state[:, 54:].reshape(x.size(0), 3, 2))
        z = trace.flatten(1)
        y = (x + self.mix(z)).tanh()
        _, h = self.gru(y.unsqueeze(1), state[:, :54].unsqueeze(0).contiguous())
        context = torch.cat((h[0], z), dim=-1)
        return context, context
