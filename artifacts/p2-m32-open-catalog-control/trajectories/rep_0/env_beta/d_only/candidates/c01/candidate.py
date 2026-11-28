import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 56, batch_first=True)
        self.decay_logit = nn.Parameter(torch.zeros(2))

    def _advance(self, x, depth, state):
        if state is None:
            hidden = x.new_zeros(x.size(0), 56)
            prior_depth = depth
            transient = torch.zeros_like(depth)
        else:
            hidden, prior_depth, transient = state

        retain = 0.05 + 0.90 * self.decay_logit.sigmoid()
        depletion = (prior_depth - depth).tanh()
        transient = retain * transient + (1 - retain) * depletion
        _, hidden = self.gru(x.unsqueeze(1), hidden.unsqueeze(0))
        hidden = hidden[0]
        context = torch.cat((hidden, transient, (0.25 * depth).tanh()), dim=-1)
        return context, (hidden, depth, transient)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        depth = obs.book_shares[..., 0].to(x.dtype).clamp_min(0).log1p()
        return self._advance(x, depth, state)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        depth = obs.book_shares[..., 0].to(x.dtype).clamp_min(0).log1p()
        if x.size(1) == 0:
            return x.new_empty(x.size(0), 0, 60)
        state = None
        contexts = []
        for t in range(x.size(1)):
            context, state = self._advance(x[:, t], depth[:, t], state)
            contexts.append(context)
        return torch.stack(contexts, dim=1)
