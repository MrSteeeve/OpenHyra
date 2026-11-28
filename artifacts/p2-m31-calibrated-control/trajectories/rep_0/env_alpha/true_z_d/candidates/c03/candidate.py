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
        self.carry = nn.Linear(3, 1)
        nn.init.zeros_(self.carry.weight)
        nn.init.zeros_(self.carry.bias)

    def _update(self, x, best, shares, state):
        if state is None:
            prior = x.new_zeros(x.size(0), 58)
        else:
            h, last = state
            side = best.new_tensor([-1, 1])
            shift = ((best - last) * side).clamp(-10, 10)
            levels = torch.arange(10, device=x.device, dtype=torch.int64)
            source = levels + shift.unsqueeze(-1)
            valid = (source >= 0) & (source < 10)
            local = h[:, :20].reshape(-1, 2, 10)
            local = local.gather(-1, source.clamp(0, 9))

            # Share the carry rule across sides, not the recurrent state.
            depth = shares.to(torch.float32).clamp(0, 1e9).log1p()
            features = torch.stack(
                (depth, depth - depth.flip((1,)),
                 shift.abs().unsqueeze(-1).expand_as(depth)),
                dim=-1,
            ).to(x.dtype) * 0.1
            gate = 1 + 0.5 * self.carry(features).squeeze(-1).tanh()
            local = torch.where(valid, local * gate, self.boundary.tanh())
            prior = torch.cat((local.flatten(1), h[:, 20:]), dim=-1)
        _, next_h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        h = next_h[0]
        return self.readout(h), (h, best)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        best = obs.book_ticks[..., 0].to(torch.int64)
        return self._update(x, best, obs.book_shares, state)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        best = obs.book_ticks[..., 0].to(torch.int64)
        state = None
        contexts = []
        for t in range(x.size(1)):
            context, state = self._update(
                x[:, t], best[:, t], obs.book_shares[:, t], state
            )
            contexts.append(context)
        if not contexts:
            return x.new_empty(x.size(0), 0, 60)
        return torch.stack(contexts, dim=1)
