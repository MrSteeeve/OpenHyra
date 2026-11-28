import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(52, 20, batch_first=True)
        self.readout = nn.Linear(72, 60)

    @staticmethod
    def _flow(q, p, old_q, old_p):
        same = (p == old_p).all(dim=-2, keepdim=True)
        scale = 1 + (q + old_q).sum(dim=-2, keepdim=True)
        change = (q - old_q) / scale
        signed = torch.stack((change[..., 0, :], -change[..., 1, :]), dim=-2)
        return (signed * same).flatten(-2)

    def _inputs(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        q = obs.book_shares.to(x.dtype).clamp_min(0)
        p = obs.book_ticks.to(torch.int64)
        return x, q, p

    def sequence_context(self, obs):
        x, q, p = self._inputs(obs)
        old_q = torch.cat((q[:, :1], q[:, :-1]), dim=1)
        old_p = torch.cat((p[:, :1], p[:, :-1]), dim=1)
        u = torch.cat((x, self._flow(q, p, old_q, old_p)), dim=-1)
        h, _ = self.gru(u)
        return self.readout(torch.cat((h, u), dim=-1)).tanh()

    def step_context(self, obs, state):
        x, q, p = self._inputs(obs)
        if state is None:
            h = x.new_zeros(x.size(0), 20)
            old_q, old_p = q, p
        else:
            h, old_q, old_p = state
        u = torch.cat((x, self._flow(q, p, old_q, old_p)), dim=-1)
        _, h = self.gru(u.unsqueeze(1), h.unsqueeze(0))
        h = h.squeeze(0)
        context = self.readout(torch.cat((h, u), dim=-1)).tanh()
        return context, (h, q, p)
