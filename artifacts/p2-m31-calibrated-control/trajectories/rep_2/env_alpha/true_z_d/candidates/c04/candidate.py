import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.local = nn.Linear(4, 2)
        self.boundary = nn.Parameter(torch.zeros(2, 10))
        self.gru = nn.GRUCell(52, 38)
        self.readout = nn.Linear(58, 60)
        self.queue = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 1))
        self.queue_readout = nn.Linear(20, 60, bias=False)
        nn.init.zeros_(self.queue_readout.weight)

    def _advance(self, x, ticks, shares, state):
        ticks = ticks.to(torch.int64)
        best = ticks[..., 0]
        fill = self.boundary.tanh().unsqueeze(0).expand(x.size(0), -1, -1)
        if state is None:
            h = x.new_zeros(x.size(0), 38)
            aligned = fill
        else:
            h, memory, old_best = state
            sign = best.new_tensor([-1, 1])
            shift = ((best - old_best) * sign).clamp(-10, 10)
            rank = torch.arange(10, device=x.device, dtype=torch.int64)
            idx = rank + shift.unsqueeze(-1)
            aligned = torch.where(
                (idx >= 0) & (idx < 10),
                memory.gather(-1, idx.clamp(0, 9)),
                fill,
            )
        volume = shares.to(x.dtype).clamp_min(0)
        depth = volume.log1p()
        distance = (ticks - best.unsqueeze(-1)).abs().to(x.dtype).log1p()
        side = x.new_tensor([-1, 1]).view(1, 2, 1).expand_as(depth)
        gate, value = self.local(
            torch.stack((depth, distance, side, aligned), dim=-1)
        ).unbind(-1)
        memory = torch.lerp(aligned, value.tanh(), gate.sigmoid())
        flat = memory.flatten(1)
        h = self.gru(torch.cat((x, flat), dim=-1), h)

        # Read current scarcity and mirrored depth contrast without a time gate.
        scarcity = (1 + volume).reciprocal()
        contrast = (depth - depth.flip((1,))).tanh()
        pressure = self.queue(torch.stack(
            (scarcity, contrast, distance, aligned), dim=-1
        )).squeeze(-1).tanh()
        context = self.readout(torch.cat((h, flat), dim=-1))
        context = context + self.queue_readout(pressure.flatten(1))
        return context, (h, memory, best)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        return self._advance(x, obs.book_ticks, obs.book_shares, state)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = None
        contexts = []
        for t in range(x.size(1)):
            context, state = self._advance(
                x[:, t], obs.book_ticks[:, t], obs.book_shares[:, t], state
            )
            contexts.append(context)
        return torch.stack(contexts, dim=1) if contexts else x.new_empty(
            x.size(0), 0, 60
        )
