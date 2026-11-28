from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.interaction = nn.Linear(75, 32)
        self.gru = nn.GRU(32, 60, batch_first=True)

    def _encode(self, obs):
        raw = self.raw_features(obs)
        base = self.encoder(raw)
        # Bounded observation-dependent modulation before recurrent mixing.
        mod = self.interaction(raw).tanh()
        return (base + base.tanh() * mod).tanh()

    def sequence_context(self, obs):
        return self.gru(self._encode(obs))[0]

    def step_context(self, obs, state):
        x = self._encode(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return h[0], h[0]
