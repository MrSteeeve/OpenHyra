import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.queue = nn.Linear(34, 2)
        self.pair = nn.Sequential(nn.Linear(4, 12), nn.Tanh(), nn.Linear(12, 2))
        self.mix = nn.Parameter(torch.tensor(-1.0))
        self.flow = nn.Linear(4, 1, bias=False)
        self.gru = nn.GRUCell(56, 16)
        self.readout = nn.Sequential(nn.Linear(72, 60), nn.Tanh(), nn.Linear(60, 60))

    def _update(self, x, ticks, shares, event, state):
        b = x.size(0)
        if state is None:
            h = x.new_zeros(b, 16)
            memory = x.new_zeros(b, 4)
            aligned = x.new_zeros(b, 2, 10)
        else:
            h, memory, old_ticks, trace = state
            w = (ticks.unsqueeze(-1) == old_ticks.unsqueeze(-2)).to(x.dtype)
            aligned = (w * trace.unsqueeze(-2)).sum(-1) / w.sum(-1).clamp_min(1)

        e = event.to(x.dtype)
        signal = self.flow(e.sign() * e.abs().log1p()).tanh()
        retain = x.new_tensor([0.55, 0.80, 0.93, 0.985])
        memory = torch.lerp(signal.expand_as(memory), memory, retain)

        live = shares > 0
        depth = shares.to(x.dtype).clamp_min(0).log1p()
        aligned = aligned * live.to(x.dtype)
        # Match offsets from the first live quote on each side.
        first = live.to(torch.int64).argmax(-1, keepdim=True)
        offset = (ticks - ticks.gather(-1, first)).abs().to(x.dtype)
        gap = (offset.unsqueeze(-1) - offset.flip((1,)).unsqueeze(-2)).abs()
        w = (1 + gap).reciprocal() * live.flip((1,)).unsqueeze(-2)
        mass = w.sum(-1, keepdim=True)
        w = w / mass.clamp_min(1e-6)
        book = torch.stack((depth, aligned), dim=-1)
        opposite = book.flip((1,))
        paired = torch.lerp(opposite, w @ opposite, self.mix.sigmoid() * (mass > 0))
        d, a = paired.unbind(-1)
        pair = torch.stack((depth + d, depth - d, aligned + a, aligned - a), dim=-1)

        side = x.new_tensor([-1, 1]).view(1, 2, 1, 1)
        local = torch.cat((
            x[:, None, None, :].expand(-1, 2, 10, -1),
            depth.unsqueeze(-1), side.expand(b, -1, 10, -1)
        ), dim=-1)
        value, gate = (self.queue(local) + self.pair(pair).tanh()).unbind(-1)
        trace = torch.lerp(aligned, value.tanh(), gate.sigmoid()) * live
        z = torch.cat((x, trace.flatten(1), memory), dim=-1)
        h = self.gru(z, h)
        return self.readout(torch.cat((z, h), dim=-1)), (h, memory, ticks, trace)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        return self._update(x, obs.book_ticks, obs.book_shares, obs.previous_event, state)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = None
        out = []
        for t in range(x.size(1)):
            context, state = self._update(
                x[:, t], obs.book_ticks[:, t], obs.book_shares[:, t],
                obs.previous_event[:, t], state
            )
            out.append(context)
        return torch.stack(out, dim=1) if out else x.new_empty(x.size(0), 0, 60)
