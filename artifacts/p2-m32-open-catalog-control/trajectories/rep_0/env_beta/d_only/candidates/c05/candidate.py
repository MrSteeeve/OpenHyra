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
        self.key = nn.Linear(32, 4)
        self.write_gate = nn.Linear(32, 1)
        self.readout = nn.Linear(44, 60)
        self.correction = nn.Linear(16, 60, bias=False)

    def _advance(self, x, state):
        if state is None:
            state = x.new_zeros(x.size(0), 60)
        base, residual = state.split((44, 16), dim=-1)
        memory = 0.98 * residual.reshape(x.size(0), 4, 4)
        innovation = 0.5 * (
            self.observed(x).tanh() - self.forecast(base).tanh()
        )
        innovation = innovation.reshape(x.size(0), 4, 4)
        key = self.key(x).tanh()
        key = key / key.square().sum(dim=-1, keepdim=True).clamp_min(1e-6).sqrt()
        column = key.unsqueeze(-1)
        delta = (innovation - memory) @ column
        beta = 0.95 * self.write_gate(x).sigmoid().unsqueeze(-1)
        residual = (memory + beta * delta * key.unsqueeze(-2)).flatten(1)
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
