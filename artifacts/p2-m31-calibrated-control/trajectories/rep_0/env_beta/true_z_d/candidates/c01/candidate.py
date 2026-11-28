import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.queue_encoder = nn.Linear(20, 8)
        self.queue_mix = nn.Linear(16, 32, bias=False)
        self.gru = nn.GRU(32, 60, batch_first=True)

    def encode(self, obs):
        base = self.encoder(self.raw_features(obs))
        ticks = obs.book_ticks.to(torch.int64)
        depth = obs.book_shares.to(base.dtype).clamp_min(0)
        gaps = (ticks - ticks[..., :1]).abs().to(base.dtype)
        gaps = torch.where(depth > 0, gaps, torch.zeros_like(gaps))
        queues = torch.cat((depth.log1p(), gaps.log1p()), dim=-1)
        sides = self.queue_encoder(queues).tanh()
        bid, ask = sides.unbind(dim=-2)
        paired = torch.cat((0.5 * (bid + ask), 0.5 * (bid - ask)), dim=-1)
        return (base + self.queue_mix(paired)).tanh()

    def sequence_context(self, obs):
        return self.gru(self.encode(obs))[0]

    def step_context(self, obs, state):
        x = self.encode(obs)
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        _, h = self.gru(x.unsqueeze(1), state.unsqueeze(0))
        return h[0], h[0]
