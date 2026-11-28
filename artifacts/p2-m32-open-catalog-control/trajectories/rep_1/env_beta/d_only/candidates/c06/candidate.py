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
        self.depth_mix = nn.Linear(5, 8)
        self.innovation_mix = nn.Linear(8, 8, bias=False)
        self.depth_route = nn.Linear(8, 32, bias=False)
        nn.init.constant_(self.innovation_gate.bias, -2.0)
        nn.init.normal_(self.route.weight, std=0.02)
        nn.init.normal_(self.depth_route.weight, std=0.02)

    def _depth_context(self, shares, dtype):
        q = shares.to(dtype).clamp_min(0)
        touch = q[..., 0]
        touch_log = touch.log1p()
        deep_log = q[..., 1:].log1p().mean(dim=-1)
        imbalance = (touch[..., :1] - touch[..., 1:]) / (
            1.0 + touch.sum(dim=-1, keepdim=True)
        )
        depth = torch.cat(
            ((touch_log / 4).tanh(), (deep_log / 4).tanh(), imbalance),
            dim=-1,
        )
        return self.depth_mix(depth).tanh()

    def _advance(self, x, event, depth, state):
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        observed = self.event_embed(event.to(x.dtype).asinh()).tanh()
        predicted = self.event_predict(state).tanh()
        innovation = self.innovation_gate(state).sigmoid() * (observed - predicted)
        x = x + self.route(innovation)
        x = x + self.depth_route(depth * self.innovation_mix(innovation))
        _, h = self.gru(x.unsqueeze(1), state.unsqueeze(0))
        return h[0]

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        depth = self._depth_context(obs.book_shares, x.dtype)
        h = self._advance(x, obs.previous_event, depth, state)
        return h, h

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        depth = self._depth_context(obs.book_shares, x.dtype)
        state = None
        contexts = []
        for t in range(x.size(1)):
            state = self._advance(x[:, t], obs.previous_event[:, t], depth[:, t], state)
            contexts.append(state)
        if not contexts:
            return x.new_empty((x.size(0), 0, 60))
        return torch.stack(contexts, dim=1)
