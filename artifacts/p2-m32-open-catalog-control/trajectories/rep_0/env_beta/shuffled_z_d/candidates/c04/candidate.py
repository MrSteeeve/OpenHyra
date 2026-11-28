import torch
from torch import nn
from torch.nn import functional as F
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.write = nn.Linear(32, 17)
        nn.init.constant_(self.write.bias[-1:], -2.0)

    def _advance(self, x, ticks, shares, state):
        b = x.size(0)
        live = shares > 0
        # Encode inactive slots outside the ordinary tick range.
        ticks = ticks.to(torch.int64)
        coords = torch.where(live, ticks, torch.iinfo(torch.int64).min)
        if state is None:
            prior = x.new_zeros((b, 60))
        else:
            memory, old_coords = state
            match = (
                (coords.unsqueeze(-1) == old_coords.unsqueeze(-2))
                & live.unsqueeze(-1)
                & (old_coords != torch.iinfo(torch.int64).min).unsqueeze(-2)
            )
            weights = match.to(x.dtype)
            aligned = (weights @ memory) / weights.sum(-1, keepdim=True).clamp_min(1)
            prior = F.pad(aligned.flatten(1), (0, 20))

        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        matrix = h[0].reshape(b, 6, 10)
        key, value, strength = self.write(x).split((10, 6, 1), dim=-1)
        key = F.normalize(key, dim=-1, eps=1e-4).unsqueeze(-1)
        error = value.tanh().unsqueeze(-1) - matrix @ key
        beta = 0.95 * strength.sigmoid().unsqueeze(-1)
        context = (matrix + beta * error * key.transpose(-1, -2)).flatten(1)
        memory = context[:, :40].reshape(b, 2, 10, 2)
        memory = memory * live.unsqueeze(-1)
        return context, (memory, coords)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        return self._advance(x, obs.book_ticks, obs.book_shares, state)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        if x.size(1) == 0:
            return x.new_empty((x.size(0), 0, 60))
        state = None
        contexts = []
        for t, current in enumerate(x.unbind(1)):
            context, state = self._advance(
                current, obs.book_ticks[:, t], obs.book_shares[:, t], state
            )
            contexts.append(context)
        return torch.stack(contexts, dim=1)
