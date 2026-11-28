import torch
from torch import nn
from torch.nn import functional as F
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.queue_encoder = nn.Linear(40, 32, bias=False)
        self.gru = nn.GRUCell(32, 20)
        self.readout = nn.Linear(52, 60)
        self.erase_key = nn.Linear(40, 20)
        self.erase_gate = nn.Linear(32, 1)
        nn.init.constant_(self.erase_gate.bias, -2.0)

    def _update(self, encoded, ticks, shares, state):
        ticks = ticks.to(torch.int64)
        depth = shares.to(encoded.dtype).clamp_min(0).log1p()
        depth = torch.where(shares > 0, depth, -torch.ones_like(depth))
        if state is None:
            hidden = encoded.new_zeros(encoded.size(0), 20)
            delta = torch.zeros_like(depth)
            matched = torch.zeros_like(depth)
        else:
            hidden, old_ticks, old_depth = state
            occupied = old_depth >= 0
            match = (
                (ticks.unsqueeze(-1) == old_ticks.unsqueeze(-2))
                & (depth.unsqueeze(-1) >= 0)
                & occupied.unsqueeze(-2)
            )
            count = match.sum(dim=-1)
            aligned = (
                match.to(depth.dtype) * old_depth.unsqueeze(-2)
            ).sum(dim=-1) / count.clamp_min(1)
            matched = (count > 0).to(depth.dtype)
            delta = (depth - aligned).clamp(-8, 8) * matched

            departed = occupied & ~match.any(dim=-2)
            mask = departed.to(depth.dtype)
            route = torch.cat(
                (mask.flatten(1),
                 (mask * old_depth.clamp(0, 8)).flatten(1)), dim=-1
            )
            key = F.normalize(self.erase_key(route), dim=-1, eps=1e-4)
            fraction = mask.sum((1, 2)) / occupied.sum((1, 2)).clamp_min(1)
            beta = 0.95 * fraction[:, None] * self.erase_gate(encoded).sigmoid()
            hidden = hidden - beta * (hidden * key).sum(-1, keepdim=True) * key

        queue = torch.cat((delta.flatten(1), matched.flatten(1)), dim=-1)
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
