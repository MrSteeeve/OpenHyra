import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.event = nn.Linear(5, 20)
        self.mix = nn.Linear(20, 32, bias=False)
        nn.init.zeros_(self.mix.weight)

    def _encode(self, obs):
        base = self.encoder(self.raw_features(obs))
        q = obs.book_shares.to(base.dtype).clamp(0, 1e8)
        depth = q.log1p()
        relative = depth - depth.mean(dim=(-2, -1), keepdim=True)
        queue = (0.5 * relative).tanh()
        event = obs.previous_event.to(base.dtype).clamp(-1e8, 1e8)
        event = event.sign() * event.abs().log1p()
        own = obs.is_own.to(base.dtype).unsqueeze(-1)
        drive = self.event(torch.cat((event, own), dim=-1)).tanh()
        return (base + self.mix(queue.flatten(-2) * drive)).tanh()

    def sequence_context(self, obs):
        return self.gru(self._encode(obs))[0]

    def step_context(self, obs, state):
        x = self._encode(obs)
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return h[0], h[0]
