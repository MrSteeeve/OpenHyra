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
        self.erase_logit = nn.Parameter(torch.tensor(-2.0))
        self.key_map = nn.Linear(3, 3)

    def _queues(self, shares, dtype):
        depth = shares.to(dtype).clamp_min(0).log1p()
        scarcity = (1.0 + depth).reciprocal()
        relative = (depth - depth.mean(dim=-1, keepdim=True)).tanh()
        queues = torch.stack(
            (depth * scarcity, scarcity, relative), dim=-1
        )
        return queues, scarcity

    def _advance(self, x, shares, prior):
        queues, scarcity = self._queues(shares, x.dtype)
        if prior is None:
            prior = x.new_zeros((x.size(0), 60))

        key = (self.key_map(queues).tanh() *
               scarcity.unsqueeze(-1)).flatten(-3)
        key = key / key.square().sum(-1, keepdim=True).clamp_min(1e-4).sqrt()
        strength = (0.95 * self.erase_logit.sigmoid() *
                    scarcity.amax(dim=(-2, -1)).unsqueeze(-1))
        routed = prior - strength * key * (key * prior).sum(-1, keepdim=True)
        _, state = self.gru(x.unsqueeze(1), routed.unsqueeze(0))
        h = state[0]

        slots = h.reshape(h.size(0), 2, 10, 3)
        local = self.local(torch.cat((queues, slots), dim=-1)).tanh()
        repair = (scarcity.unsqueeze(-1) * local).flatten(-3)
        return h + 0.25 * self.readout(repair).tanh(), h

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        return self._advance(x, obs.book_shares, state)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = None
        contexts = []
        for t in range(x.size(1)):
            context, state = self._advance(x[:, t], obs.book_shares[:, t], state)
            contexts.append(context)
        if not contexts:
            return x.new_empty((x.size(0), 0, 60))
        return torch.stack(contexts, dim=1)
