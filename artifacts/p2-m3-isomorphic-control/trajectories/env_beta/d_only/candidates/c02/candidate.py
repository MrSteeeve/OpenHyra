import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.drive = nn.Linear(32, 8)
        self.mix = nn.Parameter(torch.eye(8))
        self.rate = nn.Parameter(torch.linspace(-2.0, 0.0, 8))
        self.gru = nn.GRU(40, 52, batch_first=True)

    def _advance(self, x, state):
        if state is None:
            state = x.new_zeros(x.size(0), 60)
        h, a = state.split((52, 8), dim=-1)
        # Each row of the recurrent operator sums to less than one.
        decay = 0.98 * self.rate.sigmoid()
        carried = a @ self.mix.softmax(dim=-1).transpose(0, 1)
        a = decay * carried + (1.0 - decay) * self.drive(x).sigmoid()
        read = 2.0 * a - 1.0
        z = torch.cat((x, read), dim=-1)
        _, hn = self.gru(z.unsqueeze(1), h.unsqueeze(0).contiguous())
        return torch.cat((hn[0], read), dim=-1), torch.cat((hn[0], a), dim=-1)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = None
        outputs = []
        for xt in x.unbind(dim=1):
            ctx, state = self._advance(xt, state)
            outputs.append(ctx)
        if not outputs:
            return x.new_empty(x.size(0), 0, 60)
        return torch.stack(outputs, dim=1)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        return self._advance(x, state)
