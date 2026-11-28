import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.sweep_projection = nn.Linear(12, 32, bias=False)
        self.gru = nn.GRU(32, 60, batch_first=True)

    def _encode(self, obs):
        x = self.encoder(self.raw_features(obs))
        q = obs.book_shares.to(x.dtype).clamp_min(0)
        ticks = obs.book_ticks.to(torch.int64)
        unit = q[..., :, 0].mean(dim=-1).clamp_min(1)
        depth = (q / unit[..., None, None]).cumsum(dim=-1)
        sizes = x.new_tensor([0.5, 1.0, 2.0])
        # Crossing gap j requires clearing queues 0 through j.
        reach = torch.sigmoid(
            6.0 * (sizes - depth[..., :-1, None])
        )
        gap = (ticks[..., 1:] - ticks[..., :-1]).abs().to(x.dtype)
        # Absent queue pairs do not contribute a quoted price gap.
        pair = (q[..., 1:] > 0) & (q[..., :-1] > 0)
        displacement = (
            reach * (gap.clamp_max(64) * pair)[..., None]
        ).sum(dim=-2)
        cleared = (reach * (q[..., :-1] > 0)[..., None]).sum(dim=-2)
        sweep = torch.cat((displacement.log1p(), cleared / 9), dim=-1)
        return (x + self.sweep_projection(sweep.flatten(-2))).tanh()

    def sequence_context(self, obs):
        return self.gru(self._encode(obs))[0]

    def step_context(self, obs, state):
        x = self._encode(obs)
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        _, h = self.gru(x.unsqueeze(1), state.unsqueeze(0))
        return h[0], h[0]
