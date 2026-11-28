import torch
from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection


class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(40, 52, batch_first=True)
        self.local = nn.Linear(3, 4)
        self.mix = nn.Linear(80, 32, bias=False)
        nn.init.zeros_(self.mix.weight)
        self.flow = nn.Linear(5, 2)
        self.register_buffer("retain", torch.tensor(
            [0.60, 0.84, 0.95, 0.985]).repeat_interleave(2))

    def _encode(self, obs):
        x = self.encoder(self.raw_features(obs))
        depth = obs.book_shares.to(dtype=x.dtype).clamp_min(0).log1p()
        neighbor = torch.cat((depth[..., 1:], depth[..., -1:]), dim=-1)
        own = obs.is_own.to(dtype=x.dtype)
        q = torch.stack(
            (depth / (1 + depth), (depth - neighbor).tanh(),
             own[..., None, None].expand_as(depth)), dim=-1)
        local = self.local(q).tanh().flatten(start_dim=-3)
        x = (x * (1 + 0.5 * self.mix(local).tanh())).tanh()
        event = obs.previous_event.to(dtype=x.dtype)
        event = event.sign() * event.abs().log1p()
        drive = self.flow(torch.cat((event, own.unsqueeze(-1)), dim=-1))
        return x, drive.tanh().repeat_interleave(4, dim=-1).reshape(
            *drive.shape[:-1], 2, 4).transpose(-1, -2).flatten(-2)

    def _advance(self, drive, trace):
        r = self.retain.to(dtype=drive.dtype)
        return r * trace + (1 - r) * drive

    def sequence_context(self, obs):
        x, drive = self._encode(obs)
        if x.size(1) == 0:
            return x.new_zeros((x.size(0), 0, 60))
        trace = x.new_zeros((x.size(0), 8))
        traces = []
        for d in drive.unbind(dim=1):
            trace = self._advance(d, trace)
            traces.append(trace)
        traces = torch.stack(traces, dim=1)
        h, _ = self.gru(torch.cat((x, traces), dim=-1))
        return torch.cat((h, traces), dim=-1)

    def step_context(self, obs, state):
        x, drive = self._encode(obs)
        if state is None:
            state = x.new_zeros((x.size(0), 60))
        h, trace = state.split((52, 8), dim=-1)
        trace = self._advance(drive, trace)
        _, h = self.gru(torch.cat((x, trace), dim=-1).unsqueeze(1),
                        h.unsqueeze(0).contiguous())
        context = torch.cat((h[0], trace), dim=-1)
        return context, context
