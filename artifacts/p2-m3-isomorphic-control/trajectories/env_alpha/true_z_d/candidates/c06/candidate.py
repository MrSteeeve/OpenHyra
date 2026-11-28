from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.depth_gate = nn.Linear(10, 60)
        self.depth_readout = nn.Linear(10, 60)
        nn.init.zeros_(self.depth_gate.weight)
        nn.init.zeros_(self.depth_gate.bias)
        nn.init.zeros_(self.depth_readout.weight)
        nn.init.zeros_(self.depth_readout.bias)

    def _readout(self, obs, h):
        q = obs.book_shares.reshape(*h.shape[:-1], 10).to(h)
        q = q.clamp_min(0).log1p()
        q = q / (1 + q)
        return h + h * self.depth_gate(q).tanh() + self.depth_readout(q).tanh()

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        h, _ = self.gru(x)
        return self._readout(obs, h)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return self._readout(obs, h[0]), h[0]
