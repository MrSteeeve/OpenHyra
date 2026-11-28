import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.event_embed = nn.Linear(4, 8)
        self.event_predict = nn.Linear(60, 8)
        self.innovation_gate = nn.Linear(60, 8)
        self.route = nn.Linear(8, 32, bias=False)
        self.key = nn.Linear(32, 5)
        self.write = nn.Linear(32, 10)
        self.rewrite_gate = nn.Linear(2, 1)
        nn.init.constant_(self.innovation_gate.bias, -2.0)
        nn.init.normal_(self.route.weight, std=0.02)
        nn.init.zeros_(self.rewrite_gate.weight)
        nn.init.constant_(self.rewrite_gate.bias, -3.0)

    def _advance(self, x, event, shares, state):
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        observed = self.event_embed(event.to(x.dtype).asinh()).tanh()
        predicted = self.event_predict(state).tanh()
        gain = self.innovation_gate(state).sigmoid()
        x = x + self.route(gain * (observed - predicted))

        # The anchor keeps the key norm nonzero before normalization.
        k = torch.cat((self.key(x).tanh(), torch.ones_like(x[:, :1])), -1)
        k = (k / k.norm(dim=-1, keepdim=True)).unsqueeze(1)
        depth = shares.to(x.dtype).asinh().tanh().transpose(1, 2)
        beta = 0.95 * self.rewrite_gate(depth).sigmoid()
        memory = state.reshape(x.size(0), 10, 6)
        projection = (memory * k).sum(dim=-1, keepdim=True)
        value = self.write(x).tanh().unsqueeze(-1)
        memory = memory + beta * (value - projection) * k
        _, h = self.gru(x.unsqueeze(1), memory.flatten(1).unsqueeze(0))
        return h[0]

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        h = self._advance(x, obs.previous_event, obs.book_shares, state)
        return h, h

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = None
        contexts = []
        for t in range(x.size(1)):
            state = self._advance(
                x[:, t], obs.previous_event[:, t], obs.book_shares[:, t], state
            )
            contexts.append(state)
        if not contexts:
            return x.new_empty((x.size(0), 0, 60))
        return torch.stack(contexts, dim=1)
