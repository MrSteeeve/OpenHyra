import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.cell = nn.GRUCell(32, 52)
        self.measure = nn.Linear(32, 8)
        self.predict = nn.Linear(60, 8, bias=False)
        self.gain = nn.Linear(32, 8)
        nn.init.zeros_(self.gain.weight)
        nn.init.constant_(self.gain.bias, -2.0)
        self.retention = nn.Parameter(torch.linspace(-1.0, 1.0, 8))

    def _update(self, x, prior):
        fast = self.cell(x, prior[:, :52].contiguous())
        innovation = self.measure(x).tanh() - self.predict(prior).tanh()
        decay = 0.8 + 0.19 * self.retention.sigmoid()
        slow = decay * prior[:, 52:] + self.gain(x).sigmoid() * innovation
        return torch.cat((fast, slow), dim=-1)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        if x.size(1) == 0:
            return x.new_zeros((x.size(0), 0, 60))
        state = x.new_zeros((x.size(0), 60))
        contexts = []
        for xt in x.unbind(dim=1):
            state = self._update(xt, state)
            contexts.append(state)
        return torch.stack(contexts, dim=1)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        state = self._update(x, state)
        return state, state
