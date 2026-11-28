import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.queue_gate = nn.Linear(2, 3)
        nn.init.zeros_(self.queue_gate.weight)
        nn.init.zeros_(self.queue_gate.bias)

    def _gate(self, obs, x):
        q = obs.book_shares.to(dtype=x.dtype).clamp(0, 1e6)
        depth = q.log1p()
        local = torch.stack((depth, depth - depth[..., :1]), dim=-1)
        return (0.5 * self.queue_gate(local).tanh()).flatten(-3)

    def _update(self, x, gate, prior):
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        proposal = h[0]
        return proposal + gate * (proposal - prior)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        gate = self._gate(obs, x)
        prior = x.new_zeros(x.size(0), 60)
        outputs = []
        for t in range(x.size(1)):
            prior = self._update(x[:, t], gate[:, t], prior)
            outputs.append(prior)
        return torch.stack(outputs, dim=1) if outputs else x.new_empty(
            x.size(0), 0, 60
        )

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        prior = x.new_zeros(x.size(0), 60) if state is None else state
        context = self._update(x, self._gate(obs, x), prior)
        return context, context
