import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.local = nn.GRUCell(36, 2)
        self.global_cell = nn.GRUCell(72, 20)
        self.boundary = nn.Parameter(torch.zeros(2, 1, 2))
        self.readout = nn.Linear(60, 60)

    def _advance(self, x, ticks, shares, state):
        ticks = ticks.to(torch.int64)
        b = x.size(0)
        boundary = self.boundary.tanh().expand(b, 2, 10, 2)
        if state is None:
            prior = boundary
            g = x.new_zeros(b, 20)
        else:
            old_ticks, old, g = state
            match = ticks.unsqueeze(-1) == old_ticks.unsqueeze(-2)
            # Match within each side; unmatched queues start at the boundary.
            idx = match.to(torch.int64).argmax(dim=-1)
            aligned = old.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, 2))
            prior = torch.where(match.any(-1, keepdim=True), aligned, boundary)

        depth = shares.to(x.dtype).clamp_min(0).log1p()
        offset = (ticks - ticks[:, :, :1]).clamp(-100, 100).to(x.dtype)
        level = torch.arange(10, device=x.device, dtype=torch.int64)
        side = torch.arange(2, device=x.device, dtype=torch.int64)
        local = torch.stack((
            depth, offset / 10,
            level.to(x.dtype).view(1, 1, 10).expand_as(depth) / 9,
            side.to(x.dtype).view(1, 2, 1).expand_as(depth),
        ), dim=-1)
        z = torch.cat((x[:, None, None, :].expand(-1, 2, 10, -1), local), -1)
        h = self.local(z.reshape(-1, 36), prior.reshape(-1, 2))
        h = h.reshape(b, 2, 10, 2)
        flat = h.reshape(b, 40)
        g = self.global_cell(torch.cat((x, flat), -1), g)
        context = self.readout(torch.cat((g, flat), -1)).tanh()
        # The global summary is used in the readout, not carried as state.
        return context, (ticks, h)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        return self._advance(x, obs.book_ticks, obs.book_shares, state)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = None
        out = []
        for t in range(x.size(1)):
            c, state = self._advance(
                x[:, t], obs.book_ticks[:, t], obs.book_shares[:, t], state
            )
            out.append(c)
        return torch.stack(out, dim=1) if out else x.new_empty(x.size(0), 0, 60)
