import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.baseline = nn.GRUCell(32, 44)
        self.observed = nn.Linear(32, 16)
        self.forecast = nn.Linear(44, 16)
        self.retention = nn.Parameter(torch.linspace(-2.0, 2.0, 16))
        self.readout = nn.Linear(44, 60)
        self.correction = nn.Linear(16, 60, bias=False)
        self.erase = nn.Linear(32, 17)
        nn.init.constant_(self.erase.bias[16:], -2.0)

    def _advance(self, x, state):
        if state is None:
            state = x.new_zeros(x.size(0), 60)
        base, residual = state.split((44, 16), dim=-1)
        innovation = 0.5 * (
            self.observed(x).tanh() - self.forecast(base).tanh()
        )
        keep = 0.2 + 0.75 * self.retention.sigmoid()
        key, gate = self.erase(x).split((16, 1), dim=-1)
        key = key * torch.rsqrt(
            key.square().sum(dim=-1, keepdim=True) + 1e-4
        )
        carried = keep * residual
        projection = (carried * key).sum(dim=-1, keepdim=True)
        carried = carried - (0.95 * gate.sigmoid() * projection) * key
        residual = carried + (1.0 - keep) * innovation
        base = self.baseline(x, base.contiguous())
        context = self.readout(base).tanh() + self.correction(residual)
        return context, torch.cat((base, residual), dim=-1)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        return self._advance(x, state)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = None
        contexts = []
        for xt in x.unbind(dim=1):
            context, state = self._advance(xt, state)
            contexts.append(context)
        if not contexts:
            return x.new_empty(x.size(0), 0, 60)
        return torch.stack(contexts, dim=1)
