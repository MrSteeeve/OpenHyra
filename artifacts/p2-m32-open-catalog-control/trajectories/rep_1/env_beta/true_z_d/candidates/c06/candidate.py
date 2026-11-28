import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.event_embedding = nn.Linear(4, 4)
        self.event_forecast = nn.Linear(60, 4)
        self.innovation_projection = nn.Linear(4, 32, bias=False)
        self.innovation_gate = nn.Parameter(torch.full((32,), -2.0))
        self.write_key = nn.Linear(32, 60)
        self.write_value = nn.Linear(4, 1, bias=False)
        self.write_gate = nn.Linear(32, 1)
        nn.init.zeros_(self.write_gate.weight)
        nn.init.constant_(self.write_gate.bias, -2.0)

    def _advance(self, x, event, state):
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        forecast = self.event_forecast(state).tanh()
        observed = self.event_embedding(event).tanh()
        innovation = observed - forecast
        u = x + self.innovation_gate.sigmoid() * self.innovation_projection(innovation)

        key = self.write_key(x)
        key = key * (1.0 + key.square().sum(dim=-1, keepdim=True)).rsqrt()
        value = self.write_value(innovation).tanh()
        beta = 0.1 * self.write_gate(x).sigmoid()
        carried = (state * key).sum(dim=-1, keepdim=True)
        routed = state + beta * (value - carried) * key
        _, h = self.gru(u.unsqueeze(1), routed.unsqueeze(0))
        return h[0]

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        event = obs.previous_event.to(dtype=x.dtype).asinh()
        h = self._advance(x, event, state)
        return h, h

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        event = obs.previous_event.to(dtype=x.dtype).asinh()
        state = None
        contexts = []
        for t in range(x.size(1)):
            state = self._advance(x[:, t], event[:, t], state)
            contexts.append(state)
        if not contexts:
            return x.new_empty((x.size(0), 0, 60))
        return torch.stack(contexts, dim=1)
