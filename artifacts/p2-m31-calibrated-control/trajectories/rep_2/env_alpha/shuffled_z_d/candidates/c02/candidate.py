import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection

class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.queue = nn.Linear(3, 4)
        self.readout = nn.Linear(80, 32, bias=False)
        self.gru = nn.GRU(32, 60, batch_first=True)

    def encode(self, obs):
        base = self.encoder(self.raw_features(obs))
        ticks = obs.book_ticks
        offset = (ticks - ticks[..., :1]).to(base.dtype)
        orientation = base.new_tensor([-1., 1.]).unsqueeze(-1)
        distance = (offset * orientation).clamp(-128, 128) / 128
        shares = obs.book_shares.to(base.dtype).clamp(0, 1e8)
        features = torch.stack(
            (distance, shares.log1p(), shares.cumsum(dim=-1).log1p()),
            dim=-1,
        )
        queues = self.queue(features).tanh().flatten(-3)
        return (base + 0.1 * self.readout(queues).tanh()).tanh()

    def sequence_context(self, obs):
        return self.gru(self.encode(obs))[0]

    def step_context(self, obs, state):
        x = self.encode(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return h[0], h[0]
