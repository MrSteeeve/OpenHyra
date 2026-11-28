import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.pressure_readout = nn.Parameter(torch.zeros(60))

    def _readout(self, obs, hidden):
        ticks = obs.book_ticks
        depth = obs.book_shares[..., :, 0].to(hidden.dtype).clamp_min(0)
        bid, ask = depth.unbind(dim=-1)
        imbalance = (bid - ask) / (bid + ask).clamp_min(1)
        narrow = ticks[..., 1, 0] - ticks[..., 0, 0] == 1
        pressure = imbalance * narrow.to(hidden.dtype)
        return hidden + pressure.unsqueeze(-1) * self.pressure_readout.tanh()

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        hidden, _ = self.gru(x)
        return self._readout(obs, hidden)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, hidden = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        hidden = hidden[0]
        return self._readout(obs, hidden), hidden
