import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.thin_bias = nn.Parameter(torch.zeros(2, 10))
        self.queue_readout = nn.Linear(40, 60, bias=False)
        self.forecast_down = nn.Linear(60, 8, bias=False)
        self.forecast_up = nn.Linear(8, 32, bias=False)
        nn.init.zeros_(self.forecast_up.weight)
        self.innovation_gate = nn.Parameter(torch.full((32,), -2.0))

    def _readout(self, obs, h):
        q = obs.book_shares.to(h.dtype).clamp_min(0).log1p()
        thin = (self.thin_bias - q).sigmoid()
        ticks = obs.book_ticks.to(torch.int64)
        offset = (ticks - ticks[..., :1]).to(h.dtype)
        local = torch.cat(
            (thin.flatten(-2), (thin * (offset / 10).tanh()).flatten(-2)),
            dim=-1,
        )
        return h + self.queue_readout(local).tanh()

    def _update(self, x, prior):
        forecast = self.forecast_up(self.forecast_down(prior)).tanh()
        innovation = x - self.innovation_gate.sigmoid() * forecast
        _, h = self.gru(innovation.unsqueeze(1), prior.unsqueeze(0))
        return h[0]

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        h = self._update(x, prior)
        return self._readout(obs, h), h

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        h = x.new_zeros((x.size(0), 60))
        states = []
        for xt in x.unbind(dim=1):
            h = self._update(xt, h)
            states.append(h)
        if not states:
            return x.new_empty((x.size(0), 0, 60))
        return self._readout(obs, torch.stack(states, dim=1))
