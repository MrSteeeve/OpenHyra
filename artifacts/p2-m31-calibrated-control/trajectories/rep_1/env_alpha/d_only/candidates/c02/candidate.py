import torch
from torch import nn
from stochopia.neural.observation import StepObservation
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.flow = nn.Linear(20, 12, bias=False)
        self.gru = nn.GRU(44, 20, batch_first=True)
        self.readout = nn.Linear(64, 60)

    def _step(self, x, ticks, shares, state):
        valid = shares > 0
        depth = torch.where(valid, shares, 0).to(x.dtype).log1p()
        if state is None:
            prior = x.new_zeros(x.size(0), 20)
            delta = torch.zeros_like(depth)
        else:
            prior, old_ticks, old_depth = state
            # Match resting prices independently on each side.
            same = ticks.unsqueeze(-1) == old_ticks.unsqueeze(-2)
            match = same & valid.unsqueeze(-1) & (old_depth > 0).unsqueeze(-2)
            weight = match.to(x.dtype)
            count = weight.sum(dim=-1)
            aligned = (weight * old_depth.unsqueeze(-2)).sum(dim=-1)
            aligned = aligned / count.clamp_min(1)
            delta = torch.where(count > 0, depth - aligned, 0)
        z = torch.cat((x, self.flow(delta.flatten(1)).tanh()), dim=-1)
        _, h = self.gru(z.unsqueeze(1), prior.unsqueeze(0))
        h = h[0]
        context = self.readout(torch.cat((z, h), dim=-1)).tanh()
        return context, (h, ticks, depth)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        return self._step(x, obs.book_ticks, obs.book_shares, state)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = None
        outputs = []
        for t in range(x.size(1)):
            ctx, state = self._step(
                x[:, t], obs.book_ticks[:, t], obs.book_shares[:, t], state
            )
            outputs.append(ctx)
        return torch.stack(outputs, dim=1) if outputs else x.new_empty(x.size(0), 0, 60)
