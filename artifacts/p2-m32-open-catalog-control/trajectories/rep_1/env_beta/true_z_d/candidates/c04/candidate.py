import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.forecast = nn.Linear(60, 4, bias=False)
        self.route = nn.Linear(4, 32, bias=False)
        self.gate = nn.Parameter(torch.full((4,), -2.0))
        nn.init.zeros_(self.route.weight)

    def _advance(self, base, event, state):
        if state is None:
            state = base.new_zeros((base.size(0), 60))
        expected = self.forecast(state).tanh()
        innovation = event - self.gate.sigmoid() * expected
        x = (base + self.route(innovation)).tanh()
        _, h = self.gru(x.unsqueeze(1), state.unsqueeze(0))
        return h[0]

    def step_context(self, obs, state):
        base = self.encoder(self.raw_features(obs))
        event = obs.previous_event.to(dtype=base.dtype).tanh()
        h = self._advance(base, event, state)
        return h, h

    def sequence_context(self, obs):
        base = self.encoder(self.raw_features(obs))
        event = obs.previous_event.to(dtype=base.dtype).tanh()
        state = None
        contexts = []
        for t in range(base.size(1)):
            state = self._advance(base[:, t], event[:, t], state)
            contexts.append(state)
        if not contexts:
            return base.new_empty((base.size(0), 0, 60))
        return torch.stack(contexts, dim=1)
