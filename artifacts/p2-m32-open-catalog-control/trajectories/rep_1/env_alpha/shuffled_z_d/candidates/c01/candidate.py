import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.forecast = nn.Linear(58, 2)
        self.gain = nn.Linear(32, 2)
        self.gru = nn.GRUCell(34, 58)
        nn.init.zeros_(self.forecast.weight)
        nn.init.zeros_(self.forecast.bias)
        nn.init.zeros_(self.gain.weight)
        nn.init.constant_(self.gain.bias, -2.0)

    def _update(self, x, touch, state):
        if state is None:
            h = x.new_zeros(x.size(0), 58)
            residual = torch.zeros_like(touch)
        else:
            h, old_touch = state
            predicted = self.forecast(h).tanh()
            observed = (touch - old_touch).tanh()
            residual = self.gain(x).sigmoid() * (observed - predicted)
        h = self.gru(torch.cat((x, residual), dim=-1), h)
        return torch.cat((h, residual), dim=-1), (h, touch)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        touch = obs.book_shares[..., 0].to(x.dtype).clamp_min(0).log1p()
        return self._update(x, touch, state)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        touch = obs.book_shares[..., 0].to(x.dtype).clamp_min(0).log1p()
        state = None
        contexts = []
        for t in range(x.size(1)):
            context, state = self._update(x[:, t], touch[:, t], state)
            contexts.append(context)
        return torch.stack(contexts, dim=1) if contexts else x.new_zeros(
            x.size(0), 0, 60
        )
