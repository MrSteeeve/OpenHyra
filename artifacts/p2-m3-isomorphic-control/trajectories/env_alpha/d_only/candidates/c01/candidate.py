import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.local = nn.Sequential(nn.Linear(3, 12), nn.Tanh(),
                                   nn.Linear(12, 8))
        self.mix = nn.Linear(16, 32, bias=False)
        nn.init.zeros_(self.mix.weight)

    def inputs(self, obs):
        raw = self.raw_features(obs)
        q = obs.book_shares.reshape(*raw.shape[:-1], 10)
        q = q.to(dtype=raw.dtype).clamp_min(0).log1p()
        a, b = q[..., :5], q[..., 5:]
        q = torch.stack((a, b), dim=-2)
        near = q[..., :2].mean(dim=-1)
        far = q[..., 2:].mean(dim=-1)
        local = self.local(torch.stack(
            (q[..., 0], near - q[..., 0], far - near), dim=-1
        ))
        return (self.encoder(raw) + self.mix(local.flatten(-2))).tanh()

    def sequence_context(self, obs):
        return self.gru(self.inputs(obs))[0]

    def step_context(self, obs, state):
        x = self.inputs(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return h[0], h[0]
