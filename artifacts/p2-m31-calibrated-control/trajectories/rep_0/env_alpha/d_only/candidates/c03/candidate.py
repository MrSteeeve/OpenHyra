import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.queue = nn.Linear(34, 2)
        self.flow = nn.Linear(4, 1, bias=False)
        self.gru = nn.GRUCell(56, 16)
        self.readout = nn.Sequential(
            nn.Linear(72, 60), nn.Tanh(), nn.Linear(60, 60)
        )

    def _update(self, x, ticks, shares, event, state):
        b = x.size(0)
        if state is None:
            h = x.new_zeros(b, 16)
            memory = x.new_zeros(b, 4)
            aligned = x.new_zeros(b, 2, 10)
        else:
            h, memory, old_ticks, trace = state
            match = ticks.unsqueeze(-1) == old_ticks.unsqueeze(-2)
            weight = match.to(x.dtype)
            aligned = (weight * trace.unsqueeze(-2)).sum(-1)
            aligned = aligned / weight.sum(-1).clamp_min(1)

        e = event.to(x.dtype)
        # Share one event projection across four causal filters.
        signal = self.flow(e.sign() * e.abs().log1p()).tanh()
        retain = x.new_tensor([0.55, 0.80, 0.93, 0.985])
        memory = torch.lerp(signal.expand_as(memory), memory, retain)

        live = shares > 0
        depth = shares.to(x.dtype).clamp_min(0).log1p()
        side = x.new_tensor([-1, 1]).view(1, 2, 1, 1)
        local = torch.cat((
            x[:, None, None, :].expand(-1, 2, 10, -1),
            depth.unsqueeze(-1),
            side.expand(b, -1, 10, -1)
        ), dim=-1)
        value, gate = self.queue(local).unbind(-1)
        trace = torch.lerp(aligned, value.tanh(), gate.sigmoid())
        trace = trace * live.to(x.dtype)
        z = torch.cat((x, trace.flatten(1), memory), dim=-1)
        h = self.gru(z, h)
        context = self.readout(torch.cat((z, h), dim=-1))
        return context, (h, memory, ticks, trace)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        return self._update(
            x, obs.book_ticks, obs.book_shares, obs.previous_event, state
        )

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = None
        contexts = []
        for t in range(x.size(1)):
            context, state = self._update(
                x[:, t], obs.book_ticks[:, t], obs.book_shares[:, t],
                obs.previous_event[:, t], state
            )
            contexts.append(context)
        return torch.stack(contexts, dim=1) if contexts else x.new_empty(
            x.size(0), 0, 60
        )
