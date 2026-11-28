import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.queue = nn.Linear(40, 60, bias=False)
        self.depth = nn.Linear(20, 12)
        self.even = nn.Linear(12, 30, bias=False)
        self.odd = nn.Linear(12, 30, bias=False)
        self.forecast = nn.Linear(60, 4)
        self.gate = nn.Parameter(torch.full((4,), -2.0))
        for layer in (self.queue, self.even, self.odd, self.forecast):
            nn.init.zeros_(layer.weight)
        nn.init.zeros_(self.forecast.bias)

    def _read(self, shares, h):
        q = shares.to(dtype=h.dtype).clamp_min(0)
        log_q = q.log1p()
        local = torch.stack((log_q.tanh(), (1 + q).reciprocal()), dim=-1)
        depth = torch.cat((log_q, q.cumsum(dim=-1).log1p()), dim=-1)
        bid, ask = self.depth(depth).tanh().unbind(dim=-2)
        paired = torch.cat(
            (self.even((bid + ask) * 0.5), self.odd((bid - ask) * 0.5)),
            dim=-1,
        )
        return h + self.queue(local.flatten(-3)).tanh() + paired.tanh()

    def _update(self, x, event, prior):
        if prior is None:
            prior = x.new_zeros((x.size(0), 60))
        predicted = self.forecast(prior).tanh()
        innovation = event.to(dtype=x.dtype).tanh() - predicted
        _, hidden = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        h = hidden[0]
        # Blend the residual into four recurrent coordinates.
        front = torch.lerp(h[:, :4], innovation * 0.5, self.gate.sigmoid())
        return torch.cat((front, h[:, 4:]), dim=-1)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        if x.size(1) == 0:
            return x.new_empty((x.size(0), 0, 60))
        state = None
        states = []
        for t in range(x.size(1)):
            state = self._update(x[:, t], obs.previous_event[:, t], state)
            states.append(state)
        return self._read(obs.book_shares, torch.stack(states, dim=1))

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = self._update(x, obs.previous_event, state)
        return self._read(obs.book_shares, state), state
