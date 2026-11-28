import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.cell = nn.GRUCell(32, 52)
        self.measure = nn.Linear(32, 8)
        self.predict = nn.Linear(60, 8, bias=False)
        self.gain = nn.Linear(32, 8)
        nn.init.zeros_(self.gain.weight)
        nn.init.constant_(self.gain.bias, -2.0)
        self.retention = nn.Parameter(torch.linspace(-1.0, 1.0, 8))
        self.read = nn.Linear(52, 12, bias=False)
        self.write = nn.Linear(12, 52, bias=False)
        nn.init.zeros_(self.write.weight)

    def _inputs(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        q = obs.book_shares.to(dtype=x.dtype).reshape(*x.shape[:-1], 2, -1)
        q = q.clamp_min(0).log1p()
        touch = q[..., 0]
        inner = q[..., :3].mean(dim=-1)
        depth = torch.cat((1 / (1 + touch), (inner - touch).tanh()), dim=-1)
        return x, depth

    def _update(self, x, depth, prior):
        h = prior[:, :52].contiguous()
        proposal = self.cell(x, h)
        # Depth selects a low-rank correction to the recurrent increment.
        modes = self.read(proposal - h).tanh().reshape(-1, 4, 3)
        correction = self.write((modes * depth.unsqueeze(-1)).flatten(1))
        fast = proposal + 0.25 * correction.tanh()
        innovation = self.measure(x).tanh() - self.predict(prior).tanh()
        decay = 0.8 + 0.19 * self.retention.sigmoid()
        slow = decay * prior[:, 52:] + self.gain(x).sigmoid() * innovation
        return torch.cat((fast, slow), dim=-1)

    def sequence_context(self, obs):
        x, depth = self._inputs(obs)
        if x.size(1) == 0:
            return x.new_zeros((x.size(0), 0, 60))
        state = x.new_zeros((x.size(0), 60))
        contexts = []
        for xt, qt in zip(x.unbind(dim=1), depth.unbind(dim=1)):
            state = self._update(xt, qt, state)
            contexts.append(state)
        return torch.stack(contexts, dim=1)

    def step_context(self, obs, state):
        x, depth = self._inputs(obs)
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        state = self._update(x, depth, state)
        return state, state
