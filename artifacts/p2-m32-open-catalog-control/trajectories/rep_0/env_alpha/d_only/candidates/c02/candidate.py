import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.queue = nn.Linear(3, 12)
        self.repair = nn.Linear(24, 60, bias=False)
        nn.init.zeros_(self.repair.weight)

    def _readout(self, obs, h):
        depth = obs.book_shares.to(h.dtype).clamp_min(0).log1p()
        ticks = obs.book_ticks
        distance = (ticks - ticks[..., :1]).to(h.dtype).abs().clamp_max(32)
        level = torch.arange(10, device=h.device, dtype=torch.int64)
        rank = (level.to(h.dtype) / 9).expand_as(depth)
        local = torch.stack((depth, distance / 32, rank), dim=-1)
        encoded = self.queue(local).tanh()
        weight = (-depth - 2 * rank).softmax(dim=-1)
        sides = (weight.unsqueeze(-1) * encoded).sum(dim=-2)
        return h + 0.25 * self.repair(sides.flatten(-2)).tanh()

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        h, _ = self.gru(x)
        return self._readout(obs, h)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        state = h[0]
        return self._readout(obs, state), state
