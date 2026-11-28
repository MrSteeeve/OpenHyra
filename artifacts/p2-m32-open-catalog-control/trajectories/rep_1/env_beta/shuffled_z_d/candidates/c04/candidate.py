import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.queue_encoder = nn.Linear(40, 32, bias=False)
        self.gru = nn.GRUCell(32, 20)
        self.readout = nn.Linear(52, 60)

    def _update(self, encoded, ticks, shares, state):
        ticks = ticks.to(torch.int64)
        depth = shares.to(encoded.dtype).clamp_min(0).log1p()
        if state is None:
            hidden = encoded.new_zeros(encoded.size(0), 20)
            change = torch.zeros_like(depth)
            departure = torch.zeros_like(depth)
        else:
            hidden, old_ticks, old_depth = state
            match = (
                (ticks.unsqueeze(-1) == old_ticks.unsqueeze(-2))
                & (depth.unsqueeze(-1) > 0)
                & (old_depth.unsqueeze(-2) > 0)
            )
            weights = match.to(depth.dtype)
            aligned = (weights * old_depth.unsqueeze(-2)).sum(-1)
            aligned = aligned / match.sum(-1).clamp_min(1)
            # Current ladder: surviving deltas and newly visible depth.
            change = (depth - aligned).clamp(-8, 8)
            # Previous ladder: depth at prices no longer visible.
            departure = old_depth.clamp_max(8) * (~match.any(dim=-2))
        queue = torch.cat((change.flatten(1), departure.flatten(1)), dim=-1)
        x = (encoded + self.queue_encoder(queue)).tanh()
        hidden = self.gru(x, hidden)
        context = self.readout(torch.cat((x, hidden), dim=-1)).tanh()
        return context, (hidden, ticks, depth)

    def step_context(self, obs, state):
        return self._update(
            self.encoder(self.raw_features(obs)),
            obs.book_ticks, obs.book_shares, state,
        )

    def sequence_context(self, obs):
        encoded = self.encoder(self.raw_features(obs))
        if encoded.size(1) == 0:
            return encoded.new_empty(encoded.size(0), 0, 60)
        state = None
        contexts = []
        for t in range(encoded.size(1)):
            context, state = self._update(
                encoded[:, t], obs.book_ticks[:, t],
                obs.book_shares[:, t], state,
            )
            contexts.append(context)
        return torch.stack(contexts, dim=1)
