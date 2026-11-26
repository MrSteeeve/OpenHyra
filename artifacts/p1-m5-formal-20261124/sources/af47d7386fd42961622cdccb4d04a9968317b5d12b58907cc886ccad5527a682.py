import torch
from stochopia.neural.lob_interface import LOBWorldModel
from stochopia.neural.primitives import SeedGRULOBModel


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self):
        super().__init__()
        self.parent = SeedGRULOBModel()
        self.parent.requires_grad_(False)

        # Encoding assumed by the supplied draft: bid/ask action pairs,
        # market actions last, and side-relative offsets from -13 to +12.
        side = torch.tensor([-1, 1, -1, 1, 0, 0])
        offset = torch.arange(26) - 13
        self.register_buffer(
            "price_direction", (side[:, None] * offset).sign(), persistent=False
        )

    @staticmethod
    def _touch(observation, context):
        return (
            observation.book_ticks[..., :, 0],
            observation.book_shares[..., :, 0].to(context.dtype),
            observation.is_own.reshape(context.shape[:-1]),
        )

    @staticmethod
    def _advance(ticks, shares, own, previous):
        if previous is None:
            pressure = shares.new_zeros(shares.shape[:-1])
        else:
            old_ticks, old_shares, pressure = previous
            valid = (ticks > 0) & (old_ticks > 0)
            fraction = (old_shares - shares).clamp_min(0) / old_shares.clamp_min(1)
            retreat = torch.stack(
                (ticks[..., 0] < old_ticks[..., 0],
                 ticks[..., 1] > old_ticks[..., 1]), dim=-1
            )
            withdrawal = torch.where(
                ticks == old_ticks, fraction, retreat.to(shares.dtype)
            ).masked_fill(~valid, 0)
            bid, ask = withdrawal.unbind(-1)
            pressure = (0.85 * pressure + own.to(shares.dtype) * (ask - bid)).clamp(-1, 1)
        return ticks, shares, pressure

    def sequence_context(self, observation):
        context = self.parent.sequence_context(observation)
        ticks, shares, own = self._touch(observation, context)
        previous = None
        history = []
        for t in range(context.size(1)):
            previous = self._advance(ticks[:, t], shares[:, t], own[:, t], previous)
            history.append(previous[-1])
        pressure = torch.stack(history, dim=1) if history else shares[..., 0]
        return torch.cat((context, pressure.unsqueeze(-1)), dim=-1)

    def step_context(self, observation, state):
        hidden = None if state is None else state[0]
        context, hidden = self.parent.step_context(observation, hidden)
        touch = self._advance(
            *self._touch(observation, context), None if state is None else state[1:]
        )
        return torch.cat((context, touch[-1].unsqueeze(-1)), dim=-1), (hidden, *touch)

    def head_logits(self, context, stage, prior):
        logits = self.parent.head_logits(context[..., :-1].contiguous(), stage, prior)
        if stage == 2:
            action = prior[1].long().reshape(context.shape[:-1])
            pressure = context[..., -1:]
            direction = self.price_direction[action].to(logits.dtype)
            logits = torch.where(
                pressure == 0, logits, logits + 0.75 * pressure * direction
            )
        # Permit the training harness to call backward without unfreezing
        # the parent or adding any optimizer-visible parameters.
        if self.training and torch.is_grad_enabled() and not logits.requires_grad:
            logits.requires_grad_(True)
        return logits
