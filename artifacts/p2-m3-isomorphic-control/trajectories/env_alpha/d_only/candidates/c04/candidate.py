import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.queue = nn.Linear(1, 6)
        nn.init.zeros_(self.queue.weight)
        nn.init.zeros_(self.queue.bias)

    def _encode(self, obs):
        return self.encoder(self.raw_features(obs)).tanh()

    def _read(self, obs, h):
        # Condition each six-channel block on its current queue.
        q = obs.book_shares.reshape(*h.shape[:-1], 10, 1).to(h.dtype)
        q = q.clamp(0, 1e8).log1p() / 8
        gain = 1 + 0.5 * self.queue(q).tanh()
        return (h.unflatten(-1, (10, 6)) * gain).flatten(-2)

    def sequence_context(self, obs):
        h, _ = self.gru(self._encode(obs))
        return self._read(obs, h)

    def step_context(self, obs, state):
        x = self._encode(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        h = h[0]
        return self._read(obs, h), h
