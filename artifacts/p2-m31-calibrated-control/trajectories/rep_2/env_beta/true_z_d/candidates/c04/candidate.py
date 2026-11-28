import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.side_encoder = nn.Linear(20, 16)
        self.mix = nn.Parameter(torch.zeros(32))
        self.gru = nn.GRU(32, 60, batch_first=True)

    def _encode(self, obs):
        base = self.encoder(self.raw_features(obs))
        ticks = obs.book_ticks
        offset = (ticks - ticks[..., :1]).to(base.dtype)
        distance = torch.stack(
            (-offset[..., 0, :], offset[..., 1, :]), dim=-2
        )
        depth = obs.book_shares.to(base.dtype).clamp_min(0).cumsum(dim=-1)
        queues = torch.cat(
            (distance.clamp(0, 128).log1p(), depth.log1p()), dim=-1
        )
        side = self.side_encoder(queues).tanh()
        bid, ask = side.unbind(dim=-2)
        paired = 0.5 * torch.cat((bid + ask, bid - ask), dim=-1)
        return (base + self.mix.tanh() * paired).tanh()

    def sequence_context(self, obs):
        return self.gru(self._encode(obs))[0]

    def step_context(self, obs, state):
        x = self._encode(obs)
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        _, h = self.gru(x.unsqueeze(1), state.unsqueeze(0))
        return h[0], h[0]
