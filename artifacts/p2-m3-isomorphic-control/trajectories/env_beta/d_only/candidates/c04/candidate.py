import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.route = nn.Linear(32, 62)
        self.forecast = nn.Linear(60, 1, bias=False)
        self.gain = nn.Linear(32, 1)
        with torch.no_grad():
            self.route.weight[-1].zero_()
            self.route.bias[-1].fill_(-4.0)
            self.forecast.weight.zero_()
            self.gain.weight.zero_()
            self.gain.bias.fill_(-2.0)

    def _advance(self, x, state):
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        # Predict the learned scalar write target before consuming x.
        prediction = self.forecast(state).tanh()
        _, h = self.gru(x.unsqueeze(1), state.unsqueeze(0))
        h = h[0]
        k, v, b = self.route(x).split((60, 1, 1), dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1, eps=1e-4)
        target = v.tanh() - self.gain(x).sigmoid() * prediction
        projection = (h * k).sum(-1, keepdim=True)
        return h + (0.95 * b.sigmoid()) * (target - projection) * k

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = None
        out = []
        for xt in x.unbind(dim=1):
            state = self._advance(xt, state)
            out.append(state)
        return torch.stack(out, dim=1) if out else x.new_empty((x.size(0), 0, 60))

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = self._advance(x, state)
        return state, state
