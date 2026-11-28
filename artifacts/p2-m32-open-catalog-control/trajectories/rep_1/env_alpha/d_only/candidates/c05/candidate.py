import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.local = nn.Linear(6, 3)
        self.readout = nn.Linear(60, 60, bias=False)
        # 121 additional parameters: 27,986 total, 60 state scalars.
        self.forecast_scale = nn.Parameter(torch.ones(60))
        self.forecast_bias = nn.Parameter(torch.zeros(60))
        self.innovation_gate = nn.Parameter(torch.tensor(-2.0))

    def _context(self, obs, h, prior):
        depth = obs.book_shares.to(h.dtype).clamp_min(0).log1p()
        scarcity = (1.0 + depth).reciprocal()
        relative = (depth - depth.mean(dim=-1, keepdim=True)).tanh()
        forecast = (
            self.forecast_scale.tanh() * prior + self.forecast_bias
        ).tanh()
        centered = h - self.innovation_gate.sigmoid() * forecast
        slots = centered.reshape(*h.shape[:-1], 2, 10, 3)
        queues = torch.stack(
            (depth / (1.0 + depth), scarcity, relative), dim=-1
        )
        local = self.local(torch.cat((queues, slots), dim=-1)).tanh()
        repair = (scarcity.unsqueeze(-1) * local).flatten(-3)
        return h + 0.25 * self.readout(repair).tanh()

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        h, _ = self.gru(x)
        prior = torch.cat((torch.zeros_like(h[:, :1]), h[:, :-1]), dim=1)
        return self._context(obs, h, prior)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, final = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        h = final[0]
        return self._context(obs, h, prior), h
