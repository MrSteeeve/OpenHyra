import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.depth = nn.Sequential(nn.Linear(20, 16), nn.Tanh(),
                                   nn.Linear(16, 16), nn.Tanh())
        self.gain = nn.Parameter(torch.full((32,), 0.25))
        self.gru = nn.GRU(32, 60, batch_first=True)

    def _encode(self, obs):
        raw = self.raw_features(obs)
        ticks = obs.book_ticks.to(raw.dtype)
        shares = obs.book_shares.to(raw.dtype).clamp_min(0)
        # Express both sides in outward coordinates from their best quote.
        direction = ticks.new_tensor([-1.0, 1.0]).unsqueeze(-1)
        offset = ((ticks - ticks[..., :1]) * direction).clamp_min(0)
        offset = torch.where(shares > 0, offset, torch.zeros_like(offset))
        levels = torch.cat((offset.log1p(), shares.log1p()), dim=-1)
        bid, ask = self.depth(levels).unbind(dim=-2)
        mirrored = 0.5 * torch.cat((bid + ask, bid - ask), dim=-1)
        return (self.encoder(raw) + self.gain.tanh() * mirrored).tanh()

    def sequence_context(self, obs):
        return self.gru(self._encode(obs))[0]

    def step_context(self, obs, state):
        x = self._encode(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return h[0], h[0]
