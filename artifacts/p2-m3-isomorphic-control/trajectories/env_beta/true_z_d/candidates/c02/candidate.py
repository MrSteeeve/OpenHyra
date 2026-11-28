import math

import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.cell = nn.GRUCell(40, 52)
        self.measure = nn.Linear(32, 2)
        self.predict = nn.Linear(60, 2, bias=False)
        self.gain = nn.Linear(32, 2)
        self.read = nn.Linear(32, 8)
        nn.init.zeros_(self.gain.weight)
        nn.init.constant_(self.gain.bias, -2.0)
        nn.init.zeros_(self.read.weight)
        nn.init.zeros_(self.read.bias)
        rates = [math.exp(-1.0 / t) for t in (2, 6, 18, 54)]
        self.retention = nn.Parameter(torch.tensor([
            math.log((r - 0.5) / (0.995 - r)) for r in rates
        ]))

    def _inputs(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        depth = obs.book_shares.to(dtype=x.dtype).reshape(
            *x.shape[:-1], -1
        ).clamp_min(0).mean(dim=-1, keepdim=True)
        clock = (1.0 + depth.log1p()).reciprocal().clamp(0.25, 1.0)
        return x, clock

    def _update(self, x, clock, prior):
        decay = (0.5 + 0.495 * self.retention.sigmoid()).pow(clock)
        innovation = self.measure(x).tanh() - self.predict(prior).tanh()
        slow = prior[:, 52:].reshape(-1, 2, 4) * decay.unsqueeze(1)
        slow = slow + (self.gain(x).sigmoid() * innovation).unsqueeze(-1)
        slow = slow.flatten(1)
        read = slow * (2.0 * self.read(x).sigmoid())
        fast = self.cell(torch.cat((x, read), dim=-1),
                         prior[:, :52].contiguous())
        return torch.cat((fast, slow), dim=-1)

    def sequence_context(self, obs):
        x, clock = self._inputs(obs)
        state = x.new_zeros((x.size(0), 60))
        contexts = []
        for xt, ct in zip(x.unbind(1), clock.unbind(1)):
            state = self._update(xt, ct, state)
            contexts.append(state)
        return torch.stack(contexts, dim=1) if contexts else state.unsqueeze(1)[:, :0]

    def step_context(self, obs, state):
        x, clock = self._inputs(obs)
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        state = self._update(x, clock, state)
        return state, state
