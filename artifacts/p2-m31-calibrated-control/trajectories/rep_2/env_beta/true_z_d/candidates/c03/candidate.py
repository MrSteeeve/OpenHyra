import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.depth_logits = nn.Parameter(torch.zeros(3, 10))
        self.readout = nn.Linear(3, 60, bias=False)
        nn.init.normal_(self.readout.weight, std=0.02)

    def _input(self, obs):
        return self.encoder(self.raw_features(obs)).tanh()

    def _pressure(self, obs, dtype):
        q = obs.book_shares.to(dtype).clamp_min(0)
        depth = q @ self.depth_logits.softmax(dim=-1).transpose(0, 1)
        bid, ask = depth.unbind(dim=-2)
        imbalance = (bid - ask) / (bid + ask).clamp_min(1)
        ticks = obs.book_ticks
        spread = (ticks[..., 1, 0] - ticks[..., 0, 0]).to(dtype)
        return imbalance / spread.abs().clamp_min(1).unsqueeze(-1)

    def sequence_context(self, obs):
        h, _ = self.gru(self._input(obs))
        return h + self.readout(self._pressure(obs, h.dtype))

    def step_context(self, obs, state):
        x = self._input(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, hn = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        h = hn[0]
        return h + self.readout(self._pressure(obs, h.dtype)), h
