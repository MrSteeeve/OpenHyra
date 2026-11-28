import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.predict = nn.Linear(60, 32)
        nn.init.zeros_(self.predict.weight)
        nn.init.zeros_(self.predict.bias)
        self.innovation_gate = nn.Parameter(torch.full((32,), -2.0))

    def _advance(self, x, state):
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        expected = self.predict(state).tanh()
        innovation = x - self.innovation_gate.sigmoid() * expected
        _, h = self.gru(innovation.unsqueeze(1), state.unsqueeze(0))
        return h[0]

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        if x.size(1) == 0:
            return x.new_empty((x.size(0), 0, 60))
        state = None
        contexts = []
        for xt in x.unbind(dim=1):
            state = self._advance(xt, state)
            contexts.append(state)
        return torch.stack(contexts, dim=1)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = self._advance(x, state)
        return state, state
