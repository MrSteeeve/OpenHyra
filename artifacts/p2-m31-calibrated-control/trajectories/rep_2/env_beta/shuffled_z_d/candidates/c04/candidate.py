import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.side = nn.Sequential(nn.Linear(20, 32), nn.Tanh(), nn.Linear(32, 16))
        self.gru = nn.GRU(32, 60, batch_first=True)

    def _encode(self, obs):
        x = self.encoder(self.raw_features(obs))
        ticks = obs.book_ticks
        depth = obs.book_shares.to(x.dtype).clamp_min(0).log1p()
        offset = (ticks - ticks[..., :1]).to(x.dtype)
        outward = torch.stack((-offset[..., 0, :], offset[..., 1, :]), dim=-2)
        distance = outward.sign() * outward.abs().log1p()
        z = self.side(torch.cat((depth, distance), dim=-1)).tanh()
        bid, ask = z.unbind(dim=-2)
        shared = torch.cat(((bid + ask) * 0.5, (bid - ask) * 0.5), dim=-1)
        return (x + shared).tanh()

    def sequence_context(self, obs):
        return self.gru(self._encode(obs))[0]

    def step_context(self, obs, state):
        x = self._encode(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return h[0], h[0]
