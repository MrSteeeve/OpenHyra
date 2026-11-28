import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.retain = nn.Linear(34, 1)
        self.gru = nn.GRUCell(52, 20)

    def _update(self, x, ticks, shares, state):
        ticks = ticks.to(torch.int64)
        live = shares > 0
        depth = (shares.to(x.dtype).clamp_min(0).log1p() / 4).tanh()
        if state is None:
            global_prior = x.new_zeros(x.size(0), 20)
            carried = depth
        else:
            global_prior, old_ticks, old_trace = state
            match = (
                (ticks.unsqueeze(-1) == old_ticks.unsqueeze(-2))
                & (old_trace.unsqueeze(-2) > 0)
                & live.unsqueeze(-1)
            )
            weights = match.to(x.dtype)
            count = weights.sum(-1)
            aligned = (weights * old_trace.unsqueeze(-2)).sum(-1)
            carried = torch.where(
                count > 0, aligned / count.clamp_min(1), depth
            )

        features = torch.cat(
            (x[:, None, None, :].expand(-1, 2, 10, -1),
             depth.unsqueeze(-1), carried.unsqueeze(-1)),
            dim=-1,
        )
        keep = 0.05 + 0.90 * self.retain(features).squeeze(-1).sigmoid()
        innovation = depth - carried
        trace = torch.where(live, depth - keep * innovation, 0.0)
        delta = torch.where(live, innovation, 0.0).flatten(1)
        global_next = self.gru(torch.cat((x, delta), -1), global_prior)
        context = torch.cat((global_next, trace.flatten(1), delta), -1)
        return context, (global_next, ticks, trace)

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
        return torch.stack(contexts, dim=1) if contexts else x.new_empty(
            (x.size(0), 0, 60)
        )
