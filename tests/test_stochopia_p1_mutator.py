"""The random operator arm emits reproducible, contract-valid candidates."""

from __future__ import annotations

import pytest
import torch

from stochopia.neural.lob_interface import LOBWorldModel, _probe, verify_world_model_contract, world_model_loss
from tasks.stochopia_p1.no_llm_mutator import FAMILIES, Mutation, mutate


def _load(proposal: Mutation) -> LOBWorldModel:
    namespace: dict[str, object] = {}
    exec(compile(proposal.source, f"<{proposal.family}>", "exec"), namespace)
    candidate = namespace["CandidateLOBWorldModel"]
    assert isinstance(candidate, type) and issubclass(candidate, LOBWorldModel)
    return candidate()


@pytest.mark.parametrize("family", FAMILIES)
def test_mutation_uses_the_common_model_contract(family: str) -> None:
    variants = [mutate(seed, family=family) for seed in (0, 1, 17)]
    assert len({proposal.width for proposal in variants}) == 3
    proposal = max(variants, key=lambda variant: variant.width)
    model = _load(proposal)
    assert sum(parameter.numel() for parameter in model.parameters()) <= 28_000
    verify_world_model_contract(model, max_params=28_000, max_state_dim=128)

    batch = _probe(torch.device("cpu"), 13)
    total = world_model_loss(model.forward_sequence(batch), batch.targets).total
    total.backward()
    assert torch.isfinite(total)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters() if parameter.requires_grad)

    _, state = model.forward_step(batch.observation.at((slice(None), 0)))
    if family == "dual_stream":
        assert isinstance(state, tuple) and len(state) == 2
        assert all(stream.shape == (1, proposal.width // 2) for stream in state)
    else:
        assert isinstance(state, torch.Tensor)
        assert state.shape == (1, proposal.width)
        assert state.is_complex() == (family == "s5")


def test_seed_reproduces_the_operator_and_its_variant() -> None:
    assert mutate(103) == mutate(103)
    assert mutate(103, family="lru") == mutate(103, family="lru")
    variants = [mutate(seed) for seed in range(30)]
    assert {proposal.family for proposal in variants} == set(FAMILIES)
    assert len({proposal.width for proposal in variants}) > len(FAMILIES)
    assert len({proposal.source for proposal in variants}) == len(variants)


def test_selective_scan_conditions_the_gate_on_the_elapsed_time() -> None:
    model = _load(mutate(31, family="selective_scan"))
    obs = _probe(torch.device("cpu"), 3).observation.at((slice(None), 0))
    # Keep encoder content fixed so only the explicit time input changes.
    features = model._features(obs)
    model._features = lambda _observation: features
    previous = obs.previous_event.clone()
    previous[..., 0] = 0
    immediate, _ = model._transition(obs._replace(previous_event=previous))
    previous[..., 0] = 15
    delayed, _ = model._transition(obs._replace(previous_event=previous))
    assert torch.all(delayed < immediate)


def test_imbalance_mutation_changes_the_action_head() -> None:
    proposal = mutate(19, family="imbalance_head")
    model = _load(proposal)
    hidden = torch.ones(1, proposal.width)
    bid_heavy = torch.cat((hidden, torch.ones(1, 1)), dim=-1)
    ask_heavy = torch.cat((hidden, -torch.ones(1, 1)), dim=-1)
    previous = (torch.zeros(1, dtype=torch.long),)
    assert not torch.allclose(model.head_logits(bid_heavy, 1, previous),
                              model.head_logits(ask_heavy, 1, previous))
    torch.testing.assert_close(model.head_logits(bid_heavy, 0, ()),
                               model.head_logits(ask_heavy, 0, ()))
