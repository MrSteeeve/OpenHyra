import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 40, batch_first=True)
        self.fill = nn.Parameter(torch.zeros(2, 2))
        self.local = nn.Linear(5, 2)
        self.readout = nn.Linear(40, 60, bias=False)
        self.project = nn.Linear(40, 60)

    def _step(self, x, ticks, shares, state):
        ticks = ticks.to(torch.int64)
        b = x.size(0)
        if state is None:
            prior = x.new_zeros(b, 40)
        else:
            old, prices = state
            match = ticks.unsqueeze(-1) == prices.unsqueeze(-2)
            weights = match.to(old.dtype)
            count = weights.sum(-1, keepdim=True)
            routed = (weights / count.clamp_min(1)) @ old.reshape(b, 2, 10, 2)
            fill = self.fill.tanh().to(old.dtype)[None, :, None, :]
            prior = torch.where(count > 0, routed, fill).flatten(1)
        _, hidden = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        h = hidden[0]
        depth = shares.to(h.dtype).clamp_min(0).log1p()
        scarcity = (1 + depth).reciprocal()
        relative = (depth - depth.mean(-1, keepdim=True)).tanh()
        queues = torch.stack((1 - scarcity, scarcity, relative), -1)
        local = self.local(torch.cat((queues, h.reshape(b, 2, 10, 2)), -1))
        repair = (scarcity.unsqueeze(-1) * local.tanh()).flatten(1)
        return self.project(h) + 0.25 * self.readout(repair).tanh(), (h, ticks)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        return self._step(x, obs.book_ticks, obs.book_shares, state)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = None
        outputs = []
        for t in range(x.size(1)):
            context, state = self._step(
                x[:, t], obs.book_ticks[:, t], obs.book_shares[:, t], state
            )
            outputs.append(context)
        return torch.stack(outputs, 1) if outputs else x.new_empty(x.size(0), 0, 60)
