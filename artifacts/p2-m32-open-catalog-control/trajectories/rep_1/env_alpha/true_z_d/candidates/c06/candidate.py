import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRUCell(32, 40)
        self.fill = nn.Parameter(torch.zeros(2, 10))
        self.queue = nn.Sequential(nn.Linear(36, 16), nn.Tanh(), nn.Linear(16, 2))
        self.readout = nn.Linear(72, 60)

    def _update(self, x, ticks, shares, state):
        b = x.size(0)
        live = shares > 0
        missing = torch.iinfo(torch.int64).min
        prices = torch.where(live, ticks.to(torch.int64), missing)
        fill = self.fill.tanh().unsqueeze(0)
        if state is None:
            local = fill.expand(b, -1, -1)
            global_prior = x.new_zeros(b, 20)
        else:
            hidden, old_prices = state
            match = (
                (prices.unsqueeze(-1) == old_prices.unsqueeze(-2))
                & live.unsqueeze(-1)
                & (old_prices != missing).unsqueeze(-2)
            )
            weight = match.to(x.dtype)
            count = weight.sum(-1)
            old_local = hidden[:, :20].reshape(b, 2, 10)
            routed = (weight * old_local.unsqueeze(-2)).sum(-1)
            local = torch.where(count > 0, routed / count.clamp_min(1), fill)
            global_prior = hidden[:, 20:]

        prior = torch.cat((local.flatten(1), global_prior), dim=-1)
        global_next = self.gru(x, prior)[:, 20:]
        depth = shares.to(x.dtype).clamp(0, 1e6).log1p() / 14
        side = x.new_tensor([-1., 1.]).reshape(1, 2, 1).expand(b, 2, 10)
        rank = torch.arange(10, device=x.device, dtype=x.dtype) / 9
        rank = rank.reshape(1, 1, 10).expand(b, 2, 10)
        features = torch.cat((
            x[:, None, None, :].expand(b, 2, 10, 32),
            torch.stack((local, depth, side, rank), dim=-1),
        ), dim=-1)
        gate, write = self.queue(features).unbind(-1)
        local_next = torch.lerp(local, write.tanh(), gate.sigmoid())
        hidden = torch.cat((local_next.flatten(1), global_next), dim=-1)
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
