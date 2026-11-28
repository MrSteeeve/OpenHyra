import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRUCell(32, 20)
        self.queue = nn.Sequential(nn.Linear(4, 16), nn.Tanh(),
                                   nn.Linear(16, 2))
        self.readout = nn.Linear(60, 60)

    def _update(self, x, ticks, shares, state):
        ticks = ticks.to(torch.int64)
        live = shares > 0
        depth = shares.to(x.dtype).clamp_min(0).log1p().tanh()
        if state is None:
            h = x.new_zeros(x.size(0), 20)
            aligned = torch.zeros_like(depth)
            matched = torch.zeros_like(depth)
        else:
            h, old_ticks, memory = state
            same = ticks.unsqueeze(-1) == old_ticks.unsqueeze(-2)
            weights = same.to(x.dtype)
            count = weights.sum(-1)
            aligned = (weights * memory.unsqueeze(-2)).sum(-1)
            aligned = aligned / count.clamp_min(1)
            matched = (count > 0).to(x.dtype)

        z = torch.stack((depth, aligned, matched, live.to(x.dtype)), -1)
        gate, value = self.queue(z).unbind(-1)
        g = gate.sigmoid()
        memory = live.to(x.dtype) * ((1 - g) * aligned + g * value.tanh())
        h = self.gru(x, h)
        context = self.readout(torch.cat(
            (h, memory.flatten(1), depth.flatten(1)), -1)).tanh()
        return context, (h, ticks, memory)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        return self._update(x, obs.book_ticks, obs.book_shares, state)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        state, outputs = None, []
        for t in range(x.size(1)):
            context, state = self._update(
                x[:, t], obs.book_ticks[:, t], obs.book_shares[:, t], state)
            outputs.append(context)
        return torch.stack(outputs, 1) if outputs else x.new_empty(
            x.size(0), 0, 60)
