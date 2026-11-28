import torch
from torch import nn
from torch.nn import functional as F
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.queue_encoder = nn.Linear(2, 2)
        self.queue_readout = nn.Linear(40, 60, bias=False)
        nn.init.zeros_(self.queue_readout.weight)
        self.key = nn.Linear(20, 5)
        self.value = nn.Linear(32, 10)
        self.delta_gate = nn.Linear(32, 1)
        nn.init.zeros_(self.delta_gate.weight)
        nn.init.constant_(self.delta_gate.bias, -2.0)

    def _advance(self, x, scarcity, prior):
        if prior is None:
            prior = x.new_zeros((x.size(0), 60))
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        matrix = h[0].reshape(x.size(0), 6, 10)
        raw_key = self.key(scarcity.flatten(start_dim=-2))
        key = F.normalize(
            torch.cat((torch.ones_like(raw_key[:, :1]), raw_key), dim=-1),
            dim=-1,
        ).unsqueeze(-1)
        projected = (matrix * key).sum(dim=1)
        target = self.value(x).tanh()
        beta = 0.95 * self.delta_gate(x).sigmoid()
        corrected = matrix + key * (beta * (target - projected)).unsqueeze(1)
        return corrected.flatten(start_dim=1)

    def _readout(self, scarcity, h):
        touch = scarcity[..., :1].expand_as(scarcity)
        local = torch.stack((scarcity, touch), dim=-1)
        repair = scarcity.unsqueeze(-1) * self.queue_encoder(local).tanh()
        return h + self.queue_readout(repair.flatten(start_dim=-3))

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        scarcity = (1 + obs.book_shares.to(x.dtype).clamp_min(0)).reciprocal()
        state = self._advance(x, scarcity, state)
        return self._readout(scarcity, state), state

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        scarcity = (1 + obs.book_shares.to(x.dtype).clamp_min(0)).reciprocal()
        if x.size(1) == 0:
            return x.new_empty((x.size(0), 0, 60))
        state = None
        states = []
        for t in range(x.size(1)):
            state = self._advance(x[:, t], scarcity[:, t], state)
            states.append(state)
        return self._readout(scarcity, torch.stack(states, dim=1))
