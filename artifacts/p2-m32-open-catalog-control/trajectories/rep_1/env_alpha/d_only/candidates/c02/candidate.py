import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 48, batch_first=True)
        self.drive = nn.Linear(32, 12, bias=False)
        self.mix = nn.Linear(12, 48, bias=False)
        self.retention = nn.Parameter(torch.full((12,), 1.5))

    def _response(self, drive, prior):
        r = 0.05 + 0.94 * self.retention.sigmoid()
        return r * prior + (1.0 - r) * drive

    def _context(self, baseline, response):
        return torch.cat((baseline + self.mix(response), response), dim=-1)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        baseline, _ = self.gru(x)
        drives = self.drive(x).tanh()
        q = x.new_zeros((x.size(0), 12))
        responses = []
        for drive in drives.unbind(dim=1):
            q = self._response(drive, q)
            responses.append(q)
        return self._context(baseline, torch.stack(responses, dim=1))

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        h, q = state.split((48, 12), dim=-1)
        _, h = self.gru(x.unsqueeze(1), h.unsqueeze(0))
        h = h.squeeze(0)
        q = self._response(self.drive(x).tanh(), q)
        return self._context(h, q), torch.cat((h, q), dim=-1)
