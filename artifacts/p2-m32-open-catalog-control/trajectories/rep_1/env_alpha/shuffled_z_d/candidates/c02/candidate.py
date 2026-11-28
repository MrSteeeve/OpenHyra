import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.event_embedding = nn.Linear(4, 4)
        self.gate = nn.Linear(32, 4)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        self.gru = nn.GRUCell(36, 56)

    def _update(self, x, event, state):
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        hidden, forecast = state.split((56, 4), dim=-1)
        observed = self.event_embedding(event.to(dtype=x.dtype)).tanh()
        error = observed - forecast
        innovation = self.gate(x).sigmoid() * error
        hidden = self.gru(torch.cat((x, innovation), dim=-1), hidden)
        next_forecast = forecast + 0.0625 * error
        return (
            torch.cat((hidden, innovation), dim=-1),
            torch.cat((hidden, next_forecast), dim=-1),
        )

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        return self._update(x, obs.previous_event, state)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = None
        contexts = []
        for t in range(x.size(1)):
            context, state = self._update(x[:, t], obs.previous_event[:, t], state)
            contexts.append(context)
        if not contexts:
            return x.new_empty((x.size(0), 0, 60))
        return torch.stack(contexts, dim=1)
