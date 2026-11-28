import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.route = nn.Linear(32, 62)
        self.predict = nn.Sequential(
            nn.Linear(60, 16, bias=False),
            nn.Tanh(),
            nn.Linear(16, 32, bias=False),
        )
        with torch.no_grad():
            self.predict[-1].weight.zero_()
            self.route.weight[-1].zero_()
            self.route.bias[-1].fill_(-4.0)

    def _advance(self, x, own, state):
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        # Center the routing input, not the main recurrent input.
        prediction = self.predict(state).tanh()
        routed = torch.where(own, x, x - prediction)
        _, h = self.gru(x.unsqueeze(1), state.unsqueeze(0))
        h = h[0]
        k, v, b = self.route(routed).split((60, 1, 1), dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1, eps=1e-4)
        return h + (0.95 * b.sigmoid()) * (
            v.tanh() - (h * k).sum(-1, keepdim=True)
        ) * k

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        own = obs.is_own.reshape(x.size(0), x.size(1), 1).bool()
        state = None
        out = []
        for xt, ot in zip(x.unbind(dim=1), own.unbind(dim=1)):
            state = self._advance(xt, ot, state)
            out.append(state)
        return torch.stack(out, dim=1) if out else x.new_empty((x.size(0), 0, 60))

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        own = obs.is_own.reshape(x.size(0), 1).bool()
        state = self._advance(x, own, state)
        return state, state
