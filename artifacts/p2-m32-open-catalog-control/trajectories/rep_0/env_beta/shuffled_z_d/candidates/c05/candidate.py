import torch
from torch import nn
from torch.nn import functional as F
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.write = nn.Linear(32, 17)
        self.depth_gain = nn.Parameter(torch.zeros(()))
        nn.init.constant_(self.write.bias[-1:], -2.0)

    def _encode(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        depth = obs.book_shares.to(torch.float32).clamp(0, 1e12)
        pairs = depth[..., 0::2] + depth[..., 1::2]
        weight = 1 + self.depth_gain.sigmoid() / (1 + pairs.log1p())
        return x, weight.to(x.dtype)

    def _advance(self, x, weight, prior):
        b = x.size(0)
        if prior is None:
            prior = x.new_zeros((b, 60))
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        matrix = h[0].reshape(b, 6, 10)
        key, value, strength = self.write(x).split((10, 6, 1), dim=-1)
        routed = (key.reshape(b, 2, 5) * weight).flatten(1)
        key = F.normalize(routed, dim=-1, eps=1e-4).unsqueeze(-1)
        error = value.tanh().unsqueeze(-1) - matrix @ key
        beta = 0.95 * strength.sigmoid().unsqueeze(-1)
        return (matrix + beta * error * key.transpose(-1, -2)).flatten(1)

    def step_context(self, obs, state):
        x, weight = self._encode(obs)
        context = self._advance(x, weight, state)
        return context, context

    def sequence_context(self, obs):
        x, weight = self._encode(obs)
        if x.size(1) == 0:
            return x.new_empty((x.size(0), 0, 60))
        state = None
        contexts = []
        for current, route in zip(x.unbind(1), weight.unbind(1)):
            state = self._advance(current, route, state)
            contexts.append(state)
        return torch.stack(contexts, dim=1)
