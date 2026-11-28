import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 48, batch_first=True)
        self.local = nn.Sequential(nn.Linear(3, 8), nn.Tanh())
        self.route = nn.Linear(32 + 48 + 80 + 6, 6)
        self.readout = nn.Linear(60, 60)

    def _queues(self, obs, x):
        q = obs.book_shares.to(x.dtype).clamp_min(0).log1p()
        ticks = obs.book_ticks.to(torch.int64)
        gap = (ticks - ticks[..., :1]).abs().clamp_max(100).to(x.dtype)
        level = torch.arange(10, device=x.device, dtype=torch.int64)
        rank = (level.to(x.dtype) / 9).expand_as(q)
        local = self.local(torch.stack(
            (q / (1 + q), gap / (1 + gap), rank), dim=-1
        ))
        opening = 0.05 + 0.9 / (1 + q[..., :1])
        return local.flatten(-2), opening

    def _step(self, x, local, opening, state):
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        core, repair = state.split((48, 12), dim=-1)
        _, h = self.gru(x.unsqueeze(1), core.unsqueeze(0))
        core = h[0]
        repair = repair.reshape(x.size(0), 2, 6)
        shared = torch.cat((x, core), dim=-1).unsqueeze(1)
        proposal = self.route(torch.cat(
            (shared.expand(-1, 2, -1), local, repair), dim=-1
        )).tanh()
        repair = torch.lerp(repair, proposal, opening)
        state = torch.cat((core, repair.flatten(1)), dim=-1)
        return self.readout(state).tanh(), state

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        local, opening = self._queues(obs, x)
        return self._step(x, local, opening, state)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        local, opening = self._queues(obs, x)
        state = None
        contexts = []
        for t in range(x.size(1)):
            context, state = self._step(x[:, t], local[:, t], opening[:, t], state)
            contexts.append(context)
        return torch.stack(contexts, dim=1) if contexts else x.new_empty(
            (x.size(0), 0, 60)
        )
