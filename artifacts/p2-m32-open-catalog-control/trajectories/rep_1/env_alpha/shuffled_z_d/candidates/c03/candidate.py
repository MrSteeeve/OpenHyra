import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.drive = nn.Linear(32, 4)
        self.gru = nn.GRUCell(40, 52)
        self.register_buffer("retention", torch.tensor([0.75, 0.975]))

    def _update(self, x, state):
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        prior = state[:, 52:].reshape(x.size(0), 2, 4)
        r = self.retention.to(dtype=x.dtype).reshape(1, 2, 1)
        u = self.drive(x).tanh().unsqueeze(1)
        traces = (r * prior + (1 - r) * u).flatten(1)
        h = self.gru(torch.cat((x, traces), dim=-1), state[:, :52])
        return torch.cat((h, traces), dim=-1)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        if x.size(1) == 0:
            return x.new_empty((x.size(0), 0, 60))
        state = None
        outputs = []
        for item in x.unbind(dim=1):
            state = self._update(item, state)
            outputs.append(state)
        return torch.stack(outputs, dim=1)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        context = self._update(x, state)
        return context, context
