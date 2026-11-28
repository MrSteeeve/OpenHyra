import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 54, batch_first=True)
        self.route = nn.Linear(32, 56)
        self.drive = nn.Linear(32, 6)
        self.feedback = nn.Linear(6, 32, bias=False)
        self.register_buffer("decay", torch.tensor(
            [0.60, 0.80, 0.90, 0.95, 0.975, 0.9875]
        ))
        with torch.no_grad():
            self.feedback.weight.zero_()
            self.route.weight[-1].zero_()
            self.route.bias[-1].fill_(-4.0)

    def _advance(self, x, state):
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        h, m = state.split((54, 6), dim=-1)
        # Feed the history baseline back before incorporating this observation.
        z = x + self.feedback(m)
        _, h = self.gru(z.unsqueeze(1), h.unsqueeze(0).contiguous())
        h = h[0]
        k, v, b = self.route(z).split((54, 1, 1), dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1, eps=1e-4)
        h = h + 0.95 * b.sigmoid() * (
            v.tanh() - (h * k).sum(-1, keepdim=True)
        ) * k
        r = self.decay.to(dtype=x.dtype)
        m = r * m + (1 - r) * self.drive(x).tanh()
        return torch.cat((h, m), dim=-1)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = None
        out = []
        for xt in x.unbind(dim=1):
            state = self._advance(xt, state)
            out.append(state)
        return torch.stack(out, dim=1) if out else x.new_empty((x.size(0), 0, 60))

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = self._advance(x, state)
        return state, state
