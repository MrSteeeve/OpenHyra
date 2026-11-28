import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 40, batch_first=True)
        self.enter = nn.Parameter(torch.zeros(2, 1, 2))
        self.readout = nn.Linear(72, 60)

    def _update(self, x, ticks, shares, state):
        b = x.size(0)
        ticks = ticks.to(torch.int64)
        live = shares > 0
        fill = self.enter.tanh().unsqueeze(0).expand(b, 2, 10, 2)
        if state is None:
            prior = fill
        else:
            h, old_ticks = state
            # Current level i pulls from previous level j on the same side.
            match = (
                (ticks.unsqueeze(-1) == old_ticks.unsqueeze(-2))
                & live.unsqueeze(-1)
            )
            weight = match.to(x.dtype)
            weight = weight / weight.sum(-1, keepdim=True).clamp_min(1)
            pulled = weight @ h.reshape(b, 2, 10, 2)
            prior = torch.where(match.any(-1, keepdim=True), pulled, fill)

        _, hidden = self.gru(x.unsqueeze(1), prior.reshape(1, b, 40))
        h = hidden[0].reshape(b, 2, 10, 2)
        h = (h * live.unsqueeze(-1)).reshape(b, 40)
        context = self.readout(torch.cat((x, h), dim=-1)).tanh()
        invalid = torch.iinfo(torch.int64).min
        return context, (h, ticks.masked_fill(~live, invalid))

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        return self._update(x, obs.book_ticks, obs.book_shares, state)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = None
        contexts = []
        for t, xt in enumerate(x.unbind(dim=1)):
            context, state = self._update(
                xt, obs.book_ticks[:, t], obs.book_shares[:, t], state
            )
            contexts.append(context)
        if not contexts:
            return x.new_empty((x.size(0), 0, 60))
        return torch.stack(contexts, dim=1)
