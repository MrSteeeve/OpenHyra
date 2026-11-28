import torch
from torch import nn
from torch.nn import functional as F
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.write = nn.Linear(32, 18)
        nn.init.constant_(self.write.bias[-2:], -2.0)

    def _advance(self, x, prior):
        if prior is None:
            prior = x.new_zeros((x.size(0), 60))
        key, value, strength, erase = self.write(x).split((10, 6, 1, 1), dim=-1)
        key = F.normalize(key, dim=-1, eps=1e-4).unsqueeze(-1)
        kt = key.transpose(-1, -2)

        memory = prior.reshape(x.size(0), 6, 10)
        alpha = 0.95 * erase.sigmoid().unsqueeze(-1)
        routed = memory - alpha * ((memory @ key) @ kt)
        _, h = self.gru(x.unsqueeze(1), routed.flatten(1).unsqueeze(0))

        matrix = h[0].reshape(x.size(0), 6, 10)
        error = value.tanh().unsqueeze(-1) - matrix @ key
        beta = 0.95 * strength.sigmoid().unsqueeze(-1)
        return (matrix + beta * (error @ kt)).flatten(1)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        context = self._advance(x, state)
        return context, context

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        if x.size(1) == 0:
            return x.new_empty((x.size(0), 0, 60))
        state = None
        contexts = []
        for current in x.unbind(dim=1):
            state = self._advance(current, state)
            contexts.append(state)
        return torch.stack(contexts, dim=1)
