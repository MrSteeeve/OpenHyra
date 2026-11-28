import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.route = nn.Linear(32, 22)
        self.predict = nn.Linear(60, 6, bias=False)
        nn.init.zeros_(self.predict.weight)
        with torch.no_grad():
            self.route.bias[16:].fill_(-2.0)

    def _advance(self, x, state):
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        forecast = self.predict(state).tanh()
        _, h = self.gru(x.unsqueeze(1), state.unsqueeze(0))
        s = h[0].reshape(x.size(0), 6, 10)
        k, v, b = self.route(x).split((10, 6, 6), dim=-1)
        k = (k / (k.square().sum(-1, keepdim=True) + 1e-4).sqrt()).unsqueeze(1)
        read = (s * k).sum(-1)
        # Route the residual of a forecast made before the current input.
        innovation = v.tanh() - forecast
        delta = 0.95 * b.sigmoid() * (innovation - read)
        return (s + delta.unsqueeze(-1) * k).flatten(1)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = None
        contexts = []
        for xt in x.unbind(1):
            state = self._advance(xt, state)
            contexts.append(state)
        if not contexts:
            return x.new_empty((x.size(0), 0, 60))
        return torch.stack(contexts, dim=1)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = self._advance(x, state)
        return state, state
