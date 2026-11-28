import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRUCell(32, 40)
        self.fill = nn.Parameter(torch.zeros(2, 10))
        self.write = nn.Linear(2, 1)
        self.drive = nn.Linear(32, 20)
        self.readout = nn.Linear(72, 60)

    def _update(self, x, ticks, shares, state):
        live = shares > 0
        ticks = ticks.to(torch.int64)
        missing = torch.iinfo(torch.int64).min
        prices = torch.where(live, ticks, missing)
        fill = self.fill.tanh().unsqueeze(0)
        if state is None:
            local = fill.expand(x.size(0), -1, -1)
            matched = torch.zeros_like(live)
            global_prior = x.new_zeros(x.size(0), 20)
        else:
            hidden, old_prices = state
            match = (
                (prices.unsqueeze(-1) == old_prices.unsqueeze(-2))
                & live.unsqueeze(-1)
                & (old_prices != missing).unsqueeze(-2)
            )
            weight = match.to(x.dtype)
            count = weight.sum(-1)
            matched = count > 0
            old_local = hidden[:, :20].reshape(-1, 2, 10)
            routed = (weight * old_local.unsqueeze(-2)).sum(-1)
            local = torch.where(matched, routed / count.clamp_min(1), fill)
            global_prior = hidden[:, 20:]
        prior = torch.cat((local.flatten(1), global_prior), dim=-1)
        proposal = self.gru(x, prior)
        depth = shares.to(x.dtype).clamp(0, 1e6).log1p()
        drive = self.drive(x).reshape(-1, 2, 10)
        gate = self.write(torch.stack((depth, drive), dim=-1)).squeeze(-1)
        beta = 0.05 + 0.90 * gate.sigmoid()
        beta = torch.where(matched, beta, torch.ones_like(beta))
        updated = torch.lerp(local, proposal[:, :20].reshape(-1, 2, 10), beta)
        updated = torch.where(live, updated, fill)
        hidden = torch.cat((updated.flatten(1), proposal[:, 20:]), dim=-1)
        context = self.readout(torch.cat((hidden, x), dim=-1)).tanh()
        return context, (hidden, prices)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        return self._update(x, obs.book_ticks, obs.book_shares, state)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = None
        contexts = []
        for t in range(x.size(1)):
            context, state = self._update(
                x[:, t], obs.book_ticks[:, t], obs.book_shares[:, t], state
            )
            contexts.append(context)
        if not contexts:
            return x.new_empty(x.size(0), 0, 60)
        return torch.stack(contexts, dim=1)
