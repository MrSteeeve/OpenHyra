import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.slot = nn.Parameter(torch.zeros(2, 10, 9))
        self.shared = nn.Linear(32, 9)
        self.local = nn.Linear(1, 9, bias=False)
        self.recurrent = nn.Linear(3, 9, bias=False)
        self.reserve = nn.Linear(3, 3, bias=False)
        nn.init.zeros_(self.reserve.weight)
        self.shallow_bias = nn.Parameter(torch.zeros(3))

    def _drive(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        depth = obs.book_shares.to(dtype=x.dtype).clamp_min(0).log1p()
        drive = (
            self.shared(x)[..., None, None, :]
            + self.local(depth.unsqueeze(-1))
            + self.slot
        )
        return drive, depth

    def _update(self, drive, depth, state):
        if state is None:
            state = torch.zeros_like(drive[..., :3])
        xr, xz, xn = drive.chunk(3, dim=-1)
        hr, hz, hn = self.recurrent(state).chunk(3, dim=-1)
        reset = (xr + hr).sigmoid()
        retain = (xz + hz).sigmoid()
        proposal = (xn + reset * hn).tanh()
        state = torch.lerp(proposal, state, retain)

        shallow = (self.shallow_bias - depth[:, :, :-1, None]).sigmoid()
        delta = shallow * self.reserve(state[:, :, 1:, :]).tanh()
        context = torch.cat(
            (state[:, :, :-1, :] + delta, state[:, :, -1:, :]), dim=2
        )
        return context.flatten(-3), state

    def step_context(self, obs, state):
        drive, depth = self._drive(obs)
        return self._update(drive, depth, state)

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
