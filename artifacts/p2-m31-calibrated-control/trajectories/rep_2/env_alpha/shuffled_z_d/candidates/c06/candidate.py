import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.queue = nn.Sequential(nn.Linear(20, 24), nn.Tanh(),
                                   nn.Linear(24, 16))
        self.gain = nn.Parameter(torch.full((32,), 0.1))
        self.gru = nn.GRU(32, 60, batch_first=True)

    def encode(self, obs):
        raw = self.encoder(self.raw_features(obs))
        ticks = obs.book_ticks
        bid = ticks[..., 0, :].to(raw.dtype)
        ask = ticks[..., 1, :].to(raw.dtype)
        offset = torch.stack((bid[..., :1] - bid,
                              ask - ask[..., :1]), dim=-2)
        depth = obs.book_shares.to(raw.dtype).clamp_min(0).log1p()
        queues = self.queue(torch.cat((torch.asinh(offset), depth), dim=-1))
        bid, ask = queues.unbind(dim=-2)
        shared = torch.cat((0.5 * (bid + ask), 0.5 * (bid - ask)), dim=-1)
        return (raw + self.gain.tanh() * shared).tanh()

    def sequence_context(self, obs):
        return self.gru(self.encode(obs))[0]

    def step_context(self, obs, state):
        x = self.encode(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return h[0], h[0]
