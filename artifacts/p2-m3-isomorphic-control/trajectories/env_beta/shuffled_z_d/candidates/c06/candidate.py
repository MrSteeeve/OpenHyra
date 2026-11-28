import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(37, 55, batch_first=True)
        self.route = nn.Linear(32, 21)
        nn.init.constant_(self.route.bias[16:], -2.0)
        self.mark = nn.Linear(75, 5)
        self.cross = nn.Parameter(torch.zeros(5, 5))
        self.keep = nn.Parameter(torch.linspace(-1.0, 1.0, 5))

    def _advance(self, raw, state):
        x = self.encoder(raw).tanh()
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        h, trace = state.split((55, 5), dim=-1)

        # Event-clock update: each row of the transition sums to keep.
        keep = 0.45 + 0.50 * self.keep.sigmoid()
        cross = self.cross.softmax(dim=-1)
        trace = keep * (0.85 * trace + 0.15 * (trace @ cross.T))
        trace = trace + (1.0 - keep) * self.mark(raw).sigmoid()
        inp = torch.cat((x, trace.log1p()), dim=-1)

        _, hn = self.gru(inp.unsqueeze(1), h.unsqueeze(0))
        s = hn[0].reshape(x.size(0), 5, 11)
        k, v, b = self.route(x).split((11, 5, 5), dim=-1)
        k = (k / (k.square().sum(-1, keepdim=True) + 1e-4).sqrt()).unsqueeze(1)
        delta = 0.95 * b.sigmoid() * (v.tanh() - (s * k).sum(-1))
        h = (s + delta.unsqueeze(-1) * k).flatten(1)
        return torch.cat((h, trace), dim=-1)

    def sequence_context(self, obs):
        raw = self.raw_features(obs)
        state = None
        contexts = []
        for rt in raw.unbind(1):
            state = self._advance(rt, state)
            contexts.append(state)
        if not contexts:
            return raw.new_empty((raw.size(0), 0, 60))
        return torch.stack(contexts, dim=1)

    def step_context(self, obs, state):
        state = self._advance(self.raw_features(obs), state)
        return state, state
