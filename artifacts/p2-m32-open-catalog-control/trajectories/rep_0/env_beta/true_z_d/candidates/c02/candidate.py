import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.event_signal = nn.Linear(4, 4)
        self.forecast = nn.Linear(4, 4, bias=False)
        nn.init.eye_(self.forecast.weight)
        self.gain = nn.Parameter(torch.full((4,), -2.0))
        self.register_buffer("retention", torch.tensor([0.60, 0.82, 0.94, 0.98]))
        self.gru = nn.GRUCell(36, 56)

    def _update(self, x, event, state):
        if state is None:
            state = x.new_zeros(x.shape[0], 60)
        fast, trace = state.split([56, 4], dim=-1)
        signal = self.event_signal(event.to(x.dtype).asinh()).tanh()
        innovation = signal - self.forecast(trace).tanh()
        fast = self.gru(
            torch.cat((x, self.gain.sigmoid() * innovation), dim=-1), fast
        )
        r = self.retention.to(dtype=x.dtype)
        trace = r * trace + (1.0 - r) * signal
        return torch.cat((fast, trace), dim=-1)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = self._update(x, obs.previous_event, state)
        return state, state

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = None
        contexts = []
        for t in range(x.shape[1]):
            state = self._update(x[:, t], obs.previous_event[:, t], state)
            contexts.append(state)
        if not contexts:
            return x.new_empty(x.shape[0], 0, 60)
        return torch.stack(contexts, dim=1)
