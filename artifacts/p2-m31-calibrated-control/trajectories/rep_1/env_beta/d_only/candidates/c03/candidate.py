import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.cell = nn.GRUCell(38, 52)
        self.register_buffer("retention", torch.tensor([0.65, 0.93, 0.987]))
        self.register_buffer("weights", torch.arange(1, 11).float().reciprocal())
        self.mix = nn.Linear(60, 60)
        nn.init.eye_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    def _depth(self, shares, x):
        q = shares.to(x).clamp_min(0).log1p()
        return (q * self.weights).sum(-1) / self.weights.sum()

    def _advance(self, x, depth, state):
        if state is None:
            h = x.new_zeros(x.size(0), 52)
            trace = x.new_zeros(x.size(0), 2, 3)
            prior = depth
        else:
            h, trace, prior = state

        delta = depth - prior
        activity = delta.abs().sum(-1, keepdim=True)
        bid, ask = delta.tanh().unbind(-1)
        innovation = torch.stack((bid - ask, bid + ask), -1)
        # Advance signed memory on book activity but common memory on event count.
        clock = torch.cat((1 + activity.clamp(max=3), torch.ones_like(activity)), -1)
        keep = self.retention.to(x).pow(clock.unsqueeze(-1))
        trace = keep * trace + (1 - keep) * innovation.unsqueeze(-1)
        fast, medium, slow = trace.unbind(-1)
        flow = torch.cat((fast, medium - fast, slow - medium), -1)
        h = self.cell(torch.cat((x, flow), -1), h)
        imbalance = (depth[:, :1] - depth[:, 1:]).tanh()
        context = self.mix(torch.cat((h, flow, imbalance, activity.tanh()), -1))
        return context, (h, trace, depth)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        return self._advance(x, self._depth(obs.book_shares, x), state)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        depth = self._depth(obs.book_shares, x)
        state = None
        out = []
        for xt, qt in zip(x.unbind(1), depth.unbind(1)):
            context, state = self._advance(xt, qt, state)
            out.append(context)
        return torch.stack(out, 1) if out else x.new_empty(x.size(0), 0, 60)
