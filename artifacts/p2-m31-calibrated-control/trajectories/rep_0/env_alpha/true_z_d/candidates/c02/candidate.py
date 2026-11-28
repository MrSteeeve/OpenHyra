import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 58, batch_first=True)
        self.readout = nn.Linear(58, 60)
        self.boundary = nn.Parameter(torch.zeros(2, 10))

    def _update(self, x, best, state):
        if state is None:
            prior = x.new_zeros(x.size(0), 58)
        else:
            h, last = state
            # A current level reads the old level at the same price.
            side = best.new_tensor([-1, 1])
            shift = ((best - last) * side).clamp(-10, 10)
            levels = torch.arange(10, device=x.device, dtype=torch.int64)
            source = levels + shift.unsqueeze(-1)
            valid = (source >= 0) & (source < 10)
            local = h[:, :20].reshape(-1, 2, 10)
            local = local.gather(-1, source.clamp(0, 9))
            local = torch.where(valid, local, self.boundary.tanh())
            prior = torch.cat((local.flatten(1), h[:, 20:]), dim=-1)
        _, next_h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        h = next_h[0]
        return self.readout(h), (h, best)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        return self._update(x, obs.book_ticks[..., 0].to(torch.int64), state)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        best = obs.book_ticks[..., 0].to(torch.int64)
        state = None
        contexts = []
        for t in range(x.size(1)):
            context, state = self._update(x[:, t], best[:, t], state)
            contexts.append(context)
        if not contexts:
            return x.new_empty(x.size(0), 0, 60)
        return torch.stack(contexts, dim=1)
