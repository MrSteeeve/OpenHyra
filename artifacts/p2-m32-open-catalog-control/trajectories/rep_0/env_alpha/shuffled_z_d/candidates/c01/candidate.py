import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 54, batch_first=True)
        self.retention = nn.Parameter(torch.zeros(2))
        self.pressure_readout = nn.Linear(2, 6, bias=False)

    def _advance(self, x, ticks, shares, state):
        depth = shares.to(dtype=x.dtype).clamp_min(0).log1p()
        if state is None:
            hidden = x.new_zeros(x.size(0), 54)
            pressure = torch.zeros_like(depth)
        else:
            hidden, prior_ticks, prior_depth, prior_pressure = state
            same_queue = ticks == prior_ticks
            keep = 0.5 + 0.49 * self.retention.sigmoid()
            loss = (prior_depth - depth).tanh()
            pressure = torch.where(
                same_queue,
                keep * prior_pressure + (1.0 - keep) * loss,
                torch.zeros_like(depth),
            )
        _, hidden = self.gru(x.unsqueeze(1), hidden.unsqueeze(0))
        hidden = hidden[0]
        context = torch.cat(
            (hidden, self.pressure_readout(pressure).tanh()), dim=-1
        )
        return context, (hidden, ticks, depth, pressure)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        return self._advance(
            x, obs.book_ticks[:, :, 0], obs.book_shares[:, :, 0], state
        )

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        ticks = obs.book_ticks[..., 0]
        shares = obs.book_shares[..., 0]
        state = None
        contexts = []
        for t in range(x.size(1)):
            context, state = self._advance(x[:, t], ticks[:, t], shares[:, t], state)
            contexts.append(context)
        if not contexts:
            return x.new_empty(x.size(0), 0, 60)
        return torch.stack(contexts, dim=1)
