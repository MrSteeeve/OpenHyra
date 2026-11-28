import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(52, 60, batch_first=True)

    def _inputs(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        q = obs.book_shares.to(dtype=x.dtype).clamp_min(0)
        prefix = q.cumsum(dim=-1)
        # Keep the queue's share of side-local depth and its depth scale.
        # The GRU receives these features without the encoder bottleneck.
        share = q / (1 + prefix)
        scale = torch.log1p(prefix)
        depth = share * (1 + scale) + scale
        return torch.cat((x, depth.flatten(-2)), dim=-1)

    def sequence_context(self, obs):
        return self.gru(self._inputs(obs))[0]

    def step_context(self, obs, state):
        x = self._inputs(obs)
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        _, h = self.gru(x.unsqueeze(1), state.unsqueeze(0))
        return h[0], h[0]
