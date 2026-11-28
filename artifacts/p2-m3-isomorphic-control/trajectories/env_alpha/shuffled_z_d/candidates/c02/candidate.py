from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.local = nn.Linear(2, 12)
        self.query = nn.Linear(60, 12)
        self.readout = nn.Linear(12, 60, bias=False)
        nn.init.zeros_(self.readout.weight)

    def _context(self, obs, h):
        q = obs.book_shares.to(dtype=h.dtype).reshape(*h.shape[:-1], -1)
        depth = q.clamp_min(0).log1p()
        pos = depth.new_ones(depth.size(-1)).cumsum(0)
        pos = 2 * (pos - 0.5) / depth.size(-1) - 1
        v = self.local(
            __import__("torch").stack((depth, pos.expand_as(depth)), dim=-1)
        ).tanh()
        a = (v * self.query(h).tanh().unsqueeze(-2)).sum(-1)
        r = (a.softmax(dim=-1).unsqueeze(-1) * v).sum(-2)
        return h + self.readout(r)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        h, _ = self.gru(x)
        return self._context(obs, h)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return self._context(obs, h[0]), h[0]
