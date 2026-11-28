import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.slot = nn.Parameter(torch.zeros(2, 10, 6))
        self.shared = nn.Linear(32, 6)
        self.local = nn.Linear(1, 6, bias=False)
        self.recurrent = nn.Linear(2, 6, bias=False)
        self.change = nn.Linear(1, 6, bias=False)

    def _drive(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        depth = obs.book_shares.to(x.dtype).clamp_min(0).log1p()
        # Store relative changes only when the whole quoted grid is unchanged.
        return (
            self.shared(x)[..., None, None, :]
            + self.local(depth.unsqueeze(-1))
            + self.slot,
            depth,
        )

    def _update(self, drive, depth, state):
        if state is None:
            h = torch.zeros_like(drive[..., :2])
            prev = depth
        else:
            h, prev = state
        delta = (depth - prev).tanh() * (-depth).exp()
        xr, xz, xn = (drive + self.change(delta.unsqueeze(-1))).chunk(3, -1)
        hr, hz, hn = self.recurrent(h).chunk(3, -1)
        proposal = (xn + (xr + hr).sigmoid() * hn).tanh()
        h = torch.lerp(proposal, h, (xz + hz).sigmoid())
        context = torch.cat((h, depth.tanh().unsqueeze(-1)), -1)
        return context.flatten(-3), (h, depth)

    def step_context(self, obs, state):
        return self._update(*self._drive(obs), state)

    def sequence_context(self, obs):
        drive, depth = self._drive(obs)
        if drive.size(1) == 0:
            return drive.new_empty((drive.size(0), 0, 60))
        state = None
        contexts = []
        for d, q in zip(drive.unbind(1), depth.unbind(1)):
            context, state = self._update(d, q, state)
            contexts.append(context)
        return torch.stack(contexts, dim=1)
