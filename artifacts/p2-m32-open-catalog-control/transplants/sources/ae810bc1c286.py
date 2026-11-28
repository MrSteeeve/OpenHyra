import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.drive = nn.Linear(4, 8, bias=False)
        self.gru = nn.GRUCell(40, 52)
        self.register_buffer(
            "retention",
            torch.tensor([0.55, 0.68, 0.78, 0.86, 0.92, 0.95, 0.975, 0.988]),
        )

    def _update(self, x, drive, state):
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        h, memory = state.split((52, 8), dim=-1)
        r = self.retention.to(dtype=memory.dtype)
        memory = r * memory + (1 - r) * drive
        h = self.gru(torch.cat((x, memory), dim=-1), h)
        return torch.cat((h, memory), dim=-1)

    def _inputs(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        event = obs.previous_event.to(dtype=x.dtype)
        drive = self.drive(event.sign() * event.abs().log1p()).tanh()
        return x, drive

    def step_context(self, obs, state):
        x, drive = self._inputs(obs)
        context = self._update(x, drive, state)
        return context, context

    def sequence_context(self, obs):
        x, drive = self._inputs(obs)
        if x.size(1) == 0:
            return x.new_empty((x.size(0), 0, 60))
        state = None
        outputs = []
        for t in range(x.size(1)):
            state = self._update(x[:, t], drive[:, t], state)
            outputs.append(state)
        return torch.stack(outputs, dim=1)
