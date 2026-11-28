import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.event_embed = nn.Linear(4, 8)
        self.event_predict = nn.Linear(60, 8)
        self.innovation_gate = nn.Linear(60, 8)
        self.route = nn.Linear(8, 32, bias=False)
        nn.init.constant_(self.innovation_gate.bias, -2.0)
        nn.init.normal_(self.route.weight, std=0.02)

    def _advance(self, x, event, state):
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        observed = self.event_embed(event.to(x.dtype).asinh()).tanh()
        predicted = self.event_predict(state).tanh()
        gain = self.innovation_gate(state).sigmoid()
        x = x + self.route(gain * (observed - predicted))
        _, h = self.gru(x.unsqueeze(1), state.unsqueeze(0))
        return h[0]

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        h = self._advance(x, obs.previous_event, state)
        return h, h

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = None
        contexts = []
        for t in range(x.size(1)):
            state = self._advance(x[:, t], obs.previous_event[:, t], state)
            contexts.append(state)
        if not contexts:
            return x.new_empty((x.size(0), 0, 60))
        return torch.stack(contexts, dim=1)
