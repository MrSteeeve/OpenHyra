import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.flow = nn.Linear(4, 2)
        self.mix = nn.Linear(6, 32, bias=False)
        self.gate = nn.Parameter(torch.zeros(6))
        self.gru = nn.GRU(32, 54, batch_first=True)
        self.register_buffer("retention", torch.tensor([0.60, 0.88, 0.98]))

    def _inputs(self, obs):
        x = self.encoder(self.raw_features(obs))
        u = self.flow(torch.asinh(obs.previous_event.to(x.dtype))).tanh()
        q = obs.book_shares[..., :, 0].to(torch.float32).clamp_min(0)
        bid, ask = q.unbind(dim=-1)
        ticks = obs.book_ticks[..., :, 0]
        narrow = (ticks[..., 1] - ticks[..., 0]) == 1
        imbalance = ((bid - ask) / (bid + ask).clamp_min(1)).clamp(-1, 1)
        return x, u, (imbalance * narrow).to(x.dtype).unsqueeze(-1)

    def _trace(self, u, prior):
        r = self.retention.to(u.dtype).reshape(1, 3, 1)
        return (r * prior + (1 - r) * u.unsqueeze(1)).flatten(1)

    def _read(self, x, z, imbalance):
        return (x + self.mix(z * (1 + imbalance * self.gate.tanh()))).tanh()

    def sequence_context(self, obs):
        x, u, imbalance = self._inputs(obs)
        if x.size(1) == 0:
            return x.new_zeros(x.size(0), 0, 60)
        trace = x.new_zeros(x.size(0), 3, 2)
        history = []
        for current in u.unbind(dim=1):
            flat = self._trace(current, trace)
            history.append(flat)
            trace = flat.reshape(x.size(0), 3, 2)
        z = torch.stack(history, dim=1)
        h, _ = self.gru(self._read(x, z, imbalance))
        return torch.cat((h, z), dim=-1)

    def step_context(self, obs, state):
        x, u, imbalance = self._inputs(obs)
        if state is None:
            state = x.new_zeros(x.size(0), 60)
        z = self._trace(u, state[:, 54:].reshape(x.size(0), 3, 2))
        y = self._read(x, z, imbalance)
        _, h = self.gru(y.unsqueeze(1), state[:, :54].unsqueeze(0).contiguous())
        context = torch.cat((h[0], z), dim=-1)
        return context, context
