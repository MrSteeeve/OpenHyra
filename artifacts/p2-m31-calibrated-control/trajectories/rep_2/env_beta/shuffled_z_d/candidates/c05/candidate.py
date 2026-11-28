import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.queue_gate = nn.Linear(20, 60, bias=False)
        nn.init.zeros_(self.queue_gate.weight)

    def _readout(self, obs, h):
        depth = obs.book_shares.to(h.dtype).clamp_min(0)
        depth = depth.log1p().clamp_max(24.0)
        bid, ask = depth.unbind(dim=-2)
        queues = torch.cat((bid - ask, 0.5 * (bid + ask)), dim=-1)
        gain = 1.0 + 0.5 * self.queue_gate(queues).tanh()
        return h * gain

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        h, _ = self.gru(x)
        return self._readout(obs, h)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        state = h[0]
        return self._readout(obs, state), state
