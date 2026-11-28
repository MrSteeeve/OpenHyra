import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRUCell(32, 40)
        self.read = nn.Linear(3, 1)

    def _advance(self, x, ticks, shares, state):
        ticks = ticks.to(torch.int64)
        b = x.size(0)
        if state is None:
            aligned = x.new_zeros(b, 2, 10, 2)
        else:
            memory, old_ticks = state
            matches = ticks.unsqueeze(-1) == old_ticks.unsqueeze(-2)
            weights = matches.to(memory.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1)
            aligned = weights @ memory

        live = (shares > 0).unsqueeze(-1)
        aligned = aligned * live
        memory = self.gru(x, aligned.reshape(b, 40)).reshape(b, 2, 10, 2)
        memory = memory * live
        depth = shares.to(x.dtype).clamp(0, 1e9).log1p().unsqueeze(-1)
        local = self.read(torch.cat((memory, depth), dim=-1)).tanh()
        context = torch.cat((memory.reshape(b, 40), local.reshape(b, 20)), dim=-1)
        return context, (memory, ticks)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        return self._advance(x, obs.book_ticks, obs.book_shares, state)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        if x.size(1) == 0:
            return x.new_empty((x.size(0), 0, 60))
        state = None
        contexts = []
        for t in range(x.size(1)):
            context, state = self._advance(
                x[:, t], obs.book_ticks[:, t], obs.book_shares[:, t], state
            )
            contexts.append(context)
        return torch.stack(contexts, dim=1)
