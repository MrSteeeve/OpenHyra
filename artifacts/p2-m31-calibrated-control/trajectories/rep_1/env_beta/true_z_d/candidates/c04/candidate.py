import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)
        self.local = nn.Linear(4, 4)
        self.mix = nn.Linear(80, 32, bias=False)
        self.compare = nn.Linear(4, 4, bias=False)
        nn.init.zeros_(self.mix.weight)

    def _encode(self, obs):
        x = self.encoder(self.raw_features(obs))
        depth = obs.book_shares.to(dtype=x.dtype).clamp_min(0).log1p()
        neighbor = torch.cat((depth[..., 1:], depth[..., -1:]), dim=-1)
        own = obs.is_own.to(dtype=x.dtype)[..., None, None].expand_as(depth)
        balance = (depth - depth.flip((-2,))).tanh()
        q = torch.stack(
            (depth / (1 + depth), (depth - neighbor).tanh(), own, balance),
            dim=-1,
        )
        # Reversing the side axis reverses this paired comparison.
        local = self.local(q).tanh()
        paired = local - local.flip((-3,))
        local = local + 0.5 * self.compare(paired).tanh()
        gain = 1 + 0.5 * self.mix(local.flatten(start_dim=-3)).tanh()
        return (x * gain).tanh()

    def sequence_context(self, obs):
        return self.gru(self._encode(obs))[0]

    def step_context(self, obs, state):
        x = self._encode(obs)
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        _, h = self.gru(x.unsqueeze(1), state.unsqueeze(0))
        return h[0], h[0]
