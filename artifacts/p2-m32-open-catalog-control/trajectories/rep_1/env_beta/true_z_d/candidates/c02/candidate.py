import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.event_projection = nn.Linear(4, 6)
        self.innovation_gate = nn.Parameter(torch.full((6,), -2.0))
        self.gru = nn.GRUCell(38, 48)

    def _advance(self, x, event, state):
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        hidden, baseline, transient = state.split((48, 6, 6), dim=-1)
        event = event.to(dtype=x.dtype)
        event = event.sign() * event.abs().log1p()
        signal = self.event_projection(event).tanh()
        innovation = signal - baseline
        transient = 0.72 * transient + self.innovation_gate.sigmoid() * innovation
        baseline = 0.96 * baseline + 0.04 * signal
        hidden = self.gru(torch.cat((x, transient), dim=-1), hidden)
        return torch.cat((hidden, baseline, transient), dim=-1)

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        context = self._advance(x, obs.previous_event, state)
        return context, context

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        state = None
        contexts = []
        for t in range(x.size(1)):
            state = self._advance(x[:, t], obs.previous_event[:, t], state)
            contexts.append(state)
        if not contexts:
            return x.new_empty((x.size(0), 0, 60))
        return torch.stack(contexts, dim=1)
