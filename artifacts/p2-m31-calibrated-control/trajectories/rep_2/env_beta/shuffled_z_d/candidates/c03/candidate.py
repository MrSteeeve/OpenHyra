import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.rotation = nn.Parameter(torch.full((30,), 0.1))

    def _inputs(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        q = obs.book_shares[..., :, 0].to(x.dtype).clamp_min(0)
        bid, ask = q.unbind(-1)
        spread = obs.book_ticks[..., 1, 0] - obs.book_ticks[..., 0, 0]
        pressure = (bid - ask) / (bid + ask).clamp_min(1)
        return x, pressure * (spread == 1).to(x.dtype)

    def _advance(self, x, pressure, state):
        if state is None:
            state = x.new_zeros(x.size(0), 60)
        a, b = state.chunk(2, dim=-1)
        # Rotate paired memory channels without amplifying their norm.
        angle = pressure.unsqueeze(-1) * self.rotation.tanh()
        c, s = angle.cos(), angle.sin()
        prior = torch.cat((c * a - s * b, s * a + c * b), dim=-1)
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return h[0]

    def step_context(self, obs, state):
        x, pressure = self._inputs(obs)
        h = self._advance(x, pressure, state)
        return h, h

    def sequence_context(self, obs):
        x, pressure = self._inputs(obs)
        if x.size(1) == 0:
            return x.new_empty(x.size(0), 0, 60)
        state = None
        contexts = []
        for xt, pt in zip(x.unbind(1), pressure.unbind(1)):
            state = self._advance(xt, pt, state)
            contexts.append(state)
        return torch.stack(contexts, dim=1)
