import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Load the active parent's state_dict and supply the simulator's price decoding."""

    def __init__(self):
        super().__init__()
        self.state_dim = 48
        offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            offsets.append(offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + 8 * stage, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        self.register_buffer(
            "_direction", torch.zeros(BUCKET_COUNTS[1], BUCKET_COUNTS[2]),
            persistent=False,
        )
        self._has_direction = False
        self.requires_grad_(False)

    @torch.no_grad()
    def set_price_decoding(self, decoded_ticks, reference_ticks, is_price_action):
        """Install absolute-tick decoding: [action, bucket], [action], [action]."""
        device = self._direction.device
        decoded = torch.as_tensor(decoded_ticks, device=device)
        reference = torch.as_tensor(reference_ticks, device=device)
        priced = torch.as_tensor(is_price_action, device=device, dtype=torch.bool)
        actions = self._direction.shape[0]
        if (decoded.shape != self._direction.shape
                or reference.shape != (actions,) or priced.shape != (actions,)):
            raise ValueError("Expected decoded [action, bucket] and action-level metadata")
        reference = reference[:, None]
        direction = (decoded > reference).to(self._direction.dtype)
        direction -= (decoded < reference).to(direction.dtype)
        self._direction.copy_(direction.masked_fill(~priced[:, None], 0))
        self._has_direction = True

    def _features(self, observation):
        ticks = observation.book_ticks
        shares = observation.book_shares.to(self.input_projection.weight.dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(-1) / present.sum(-1).clamp_min(1)
        price = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        price = price.masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()
        bid, ask = touch.unbind(-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(-1), 0).log1p()
        bd, ad = shares.sum(-1).unbind(-1)
        imbalance = (bd - ad) / (bd + ad).clamp_min(1)
        own = observation.is_own.reshape_as(spread)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        x = torch.cat((price.flatten(-2), depth.flatten(-2),
                       torch.stack((spread, imbalance, own), -1),
                       previous.flatten(-2)), -1)
        return self.input_projection(x.to(self.input_projection.weight.dtype)).tanh()

    @staticmethod
    def _pressure(ticks, shares, own, previous):
        if previous is None:
            return shares.new_zeros(shares.shape[:-1])
        old_ticks, old_shares, old_pressure = previous
        valid = (ticks > 0) & (old_ticks > 0)
        same = ticks == old_ticks
        depleted = (old_shares - shares).clamp_min(0) / old_shares.clamp_min(1)
        away = torch.stack((ticks[..., 0] < old_ticks[..., 0],
                            ticks[..., 1] > old_ticks[..., 1]), -1)
        withdrawal = torch.where(same, depleted, away.to(depleted.dtype))
        withdrawal = withdrawal.masked_fill(~valid, 0)
        bid, ask = withdrawal.unbind(-1)
        return (0.85 * old_pressure + own * (ask - bid)).clamp(-1, 1)

    def _run(self, observation, state):
        hidden, previous = (None, None) if state is None else (state[0], state[1:])
        out, hidden = self.gru(
            self._features(observation), None if hidden is None else hidden.unsqueeze(0)
        )
        ticks = observation.book_ticks[..., 0]
        shares = observation.book_shares[..., 0].to(torch.float64)
        own = observation.is_own.reshape(ticks.shape[:-1]).to(torch.bool)
        pressures = []
        for t in range(ticks.shape[1]):
            price, size = ticks[:, t], shares[:, t]
            pressure = self._pressure(price, size, own[:, t], previous)
            previous = (price, size, pressure)
            pressures.append(pressure)
        if not pressures:
            return torch.cat((out, out[..., :1]), -1), state
        pressure = torch.stack(pressures, 1).to(out.dtype)
        return torch.cat((out, pressure.unsqueeze(-1)), -1), (hidden[0], *previous)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        return self._run(observation, None)[0]

    def step_context(self, observation: StepObservation, state: State):
        def time_axis(x):
            return None if x is None else x.unsqueeze(1)

        sequence = StepObservation(
            time_axis(observation.book_ticks), time_axis(observation.book_shares),
            time_axis(observation.previous_event), time_axis(observation.is_own),
            time_axis(observation.has_deep),
        )
        context, state = self._run(sequence, state)
        return context[:, 0], state

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        x, pressure = context[..., :-1].contiguous(), context[..., -1:]
        if prior:
            tokens = torch.stack(prior, -1) + self.token_offsets[:stage]
            x = torch.cat((x, self.token_embed(tokens).flatten(-2)), -1)
        logits = self.heads[stage](x)
        if stage == 2:
            if self._has_direction:
                logits = logits + 0.75 * pressure * self._direction[prior[1]]
            elif torch.any(pressure != 0):
                raise RuntimeError("Install the simulator's price decoding before own-order rollout")
        return logits
