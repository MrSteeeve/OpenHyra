import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.queue = nn.Linear(40, 60, bias=False)
        self.depth = nn.Linear(20, 12)
        self.even = nn.Linear(12, 30, bias=False)
        self.odd = nn.Linear(12, 30, bias=False)
        nn.init.zeros_(self.queue.weight)
        nn.init.zeros_(self.even.weight)
        nn.init.zeros_(self.odd.weight)

    def _read(self, obs, h):
        q = obs.book_shares.to(dtype=h.dtype).clamp_min(0)
        log_q = q.log1p()
        local = torch.stack((log_q.tanh(), (1 + q).reciprocal()), dim=-1)

        # Measure each side outward from its best occupied quote.
        ticks = obs.book_ticks.to(torch.int64)
        first = (q > 0).to(torch.int64).argmax(dim=-1, keepdim=True)
        offsets = (ticks - ticks.gather(-1, first)).abs()
        radii = torch.arange(10, device=q.device, dtype=torch.int64)
        at_radius = offsets.unsqueeze(-1) == radii
        within_radius = offsets.unsqueeze(-1) <= radii
        mass = q.unsqueeze(-1)
        shell = (mass * at_radius).sum(dim=-2).log1p()
        cumulative = (mass * within_radius).sum(dim=-2).log1p()
        depth = torch.cat((shell, cumulative), dim=-1)

        bid, ask = self.depth(depth).tanh().unbind(dim=-2)
        paired = torch.cat(
            (self.even((bid + ask) * 0.5), self.odd((bid - ask) * 0.5)),
            dim=-1,
        )
        return h + self.queue(local.flatten(-3)).tanh() + paired.tanh()

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        h, _ = self.gru(x)
        return self._read(obs, h)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return self._read(obs, h[0]), h[0]
