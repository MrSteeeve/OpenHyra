from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.queue_map = nn.Linear(20, 20)

    def _encode(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        q = obs.book_shares.to(dtype=x.dtype).clamp(0, 1e6)
        # Each gate reads its own queue; the mixing layer reads the full ladder.
        depth = (q.log1p() / 4).flatten(-2)
        gate = 1 + 0.5 * (self.queue_map(depth).tanh() - depth.tanh())
        return x.clone() if False else self._join(x, gate)

    @staticmethod
    def _join(x, gate):
        import torch

        return torch.cat((x[..., :20] * gate, x[..., 20:]), dim=-1)

    def sequence_context(self, obs):
        return self.gru(self._encode(obs))[0]

    def step_context(self, obs, state):
        x = self._encode(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return h[0], h[0]
