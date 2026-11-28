from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.depth_gate = nn.Linear(10, 60, bias=False)
        nn.init.zeros_(self.depth_gate.weight)

    def _readout(self, obs, h):
        q = obs.book_shares.to(dtype=h.dtype).reshape(*h.shape[:-1], 10)
        q = q.clamp_min(0)
        depth = q.log1p()
        depth = depth / (1 + depth)
        return h * (1 + self.depth_gate(depth).tanh())

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        h, _ = self.gru(x)
        return self._readout(obs, h)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        _, h = self.gru(x.unsqueeze(1), state.unsqueeze(0))
        h = h[0]
        return self._readout(obs, h), h
