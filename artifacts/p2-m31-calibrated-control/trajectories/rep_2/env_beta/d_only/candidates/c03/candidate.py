import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.drive = nn.Linear(7, 2)
        self.predict = nn.Linear(54, 2)
        self.gate = nn.Parameter(torch.full((2,), -2.0))
        self.gru = nn.GRU(38, 54, batch_first=True)

    def _inputs(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        q = obs.book_shares.to(x.dtype).clamp_min(0).log1p()
        near = q[..., :3].mean(dim=-1)
        deep = q[..., 3:].mean(dim=-1)
        p = (near[..., 0] - near[..., 1]).unsqueeze(-1)
        d = (deep[..., 0] - deep[..., 1]).unsqueeze(-1)
        u = torch.cat(
            (obs.previous_event.to(x.dtype),
             obs.is_own.to(x.dtype).unsqueeze(-1), p, d), dim=-1
        )
        return x, self.drive(u).tanh()

    def _advance(self, x, u, state):
        if state is None:
            h = x.new_zeros(x.size(0), 54)
            m = x.new_zeros(x.size(0), 3, 2)
        else:
            h, m = state
        # Predict from the preceding state; keep the fast inputs unchanged.
        pred = self.predict(h).tanh()
        slow = u - self.gate.sigmoid() * pred
        v = torch.stack((u, u, slow), dim=1)
        r = u.new_tensor((0.65, 0.90, 0.98)).reshape(1, 3, 1)
        m = r * m + (1 - r) * v
        z = m.flatten(1)
        _, hn = self.gru(
            torch.cat((x, z), dim=-1).unsqueeze(1), h.unsqueeze(0)
        )
        h = hn[0]
        return torch.cat((h, z), dim=-1), (h, m)

    def step_context(self, obs, state):
        x, u = self._inputs(obs)
        return self._advance(x, u, state)

    def sequence_context(self, obs):
        x, u = self._inputs(obs)
        if x.size(1) == 0:
            return x.new_empty(x.size(0), 0, 60)
        state = None
        contexts = []
        for xt, ut in zip(x.unbind(dim=1), u.unbind(dim=1)):
            ctx, state = self._advance(xt, ut, state)
            contexts.append(ctx)
        return torch.stack(contexts, dim=1)
