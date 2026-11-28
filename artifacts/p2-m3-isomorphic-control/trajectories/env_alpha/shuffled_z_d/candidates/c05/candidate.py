import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.depth_key = nn.Linear(10, 16, bias=False)
        self.history_key = nn.Linear(60, 16)
        self.mix = nn.Linear(16, 60, bias=False)
        nn.init.zeros_(self.mix.weight)

    def _encode(self, obs):
        return self.encoder(self.raw_features(obs)).tanh()

    def _readout(self, obs, h):
        q = obs.book_shares.reshape(*h.shape[:-1], 10).to(h.dtype)
        q = q.clamp_min(0).log1p()
        # Preserve both absolute depth and depth relative to other queues.
        depth = q / (1.0 + q.mean(dim=-1, keepdim=True))
        interaction = self.depth_key(depth).tanh() * self.history_key(h).tanh()
        return h + self.mix(interaction).tanh()

    def sequence_context(self, obs):
        h, _ = self.gru(self._encode(obs))
        return self._readout(obs, h)

    def step_context(self, obs, state):
        x = self._encode(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, next_state = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        h = next_state[0]
        return self._readout(obs, h), h
