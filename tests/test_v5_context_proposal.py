import json
from types import SimpleNamespace
import pytest

from harness import (
    _build_v5_prompt_section,
    _inject_v5_context,
    _inject_v5_proposal,
    _build_experiment_plan,
)
from harness_v5 import V5Bridge
from schemas_v5 import ExperimentEvent, ExperimentPlan
from eb import ExperienceBank
from context_agent import ContextDecision, build_inspiration


class TestBuildV5PromptSection:
    def test_renders_portfolio_and_analysis(self):
        v5_context = {
            'portfolio_text': 'Island 0 has 3 candidates, best score 0.45',
            'analysis_text': 'Island 0 shows improvement trend',
        }
        section = _build_v5_prompt_section(v5_context)
        assert '## V5 Portfolio Context' in section
        assert 'Island 0 has 3 candidates' in section
        assert '## V5 Island Analysis' in section
        assert 'improvement trend' in section

    def test_renders_portfolio_only_when_no_analysis(self):
        v5_context = {
            'portfolio_text': 'Portfolio data here',
            'analysis_text': '',
        }
        section = _build_v5_prompt_section(v5_context)
        assert '## V5 Portfolio Context' in section
        assert '## V5 Island Analysis' not in section

    def test_returns_empty_when_no_content(self):
        v5_context = {'portfolio_text': '', 'analysis_text': ''}
        assert _build_v5_prompt_section(v5_context) == ''


class TestInjectV5Context:
    def test_inserts_before_assignment_marker(self):
        prompt = 'Some context.\n\n## Your assignment\n\nDo the thing.'
        section = '## V5 Portfolio Context\n\nIsland data'
        result = _inject_v5_context(prompt, section)
        assert result.index('V5 Portfolio Context') < result.index('Your assignment')
        assert 'Some context.' in result
        assert 'Do the thing.' in result

    def test_appends_when_no_marker(self):
        prompt = 'No marker here.'
        section = '## V5 Portfolio Context\n\nIsland data'
        result = _inject_v5_context(prompt, section)
        assert result.endswith('Island data')

    def test_noop_when_section_empty(self):
        prompt = '## Your assignment\n\nStuff'
        assert _inject_v5_context(prompt, '') == prompt

    def test_clips_to_total_prompt_budget(self):
        result = _inject_v5_context(
            'prefix',
            '## V5 Portfolio Context\n\n' + ('x' * 500),
            max_total_chars=100,
        )
        assert len(result) <= 100
        assert result.startswith('prefix')


class TestInjectV5Proposal:
    def test_inserts_before_identity_marker(self):
        prompt = 'Base prompt.\n\n## Local candidate identity\n\nCandidate 1 of 4'
        result = _inject_v5_proposal(prompt, 'Plan: mutate feature IR')
        assert result.index('V5 Proposal Context') < result.index('Local candidate identity')
        assert 'Plan: mutate feature IR' in result

    def test_appends_when_no_marker(self):
        prompt = 'No marker.'
        result = _inject_v5_proposal(prompt, 'Plan text')
        assert '## V5 Proposal Context' in result
        assert result.endswith('Plan text')

    def test_noop_when_text_empty(self):
        prompt = '## Local candidate identity\n\nStuff'
        assert _inject_v5_proposal(prompt, '') == prompt

    def test_clips_to_total_prompt_budget(self):
        prompt = 'prefix\n\n## Local candidate identity\n\nCandidate'
        result = _inject_v5_proposal(prompt, 'x' * 500, max_total_chars=120)
        assert len(result) <= 120


class TestBuildExperimentPlan:
    class FakeDecision:
        action = 'continue'
        success_criterion = 'score > 0.5'

    class FakeTask:
        timeout_s = 300
        max_artifact_bytes = 1_000_000
        engineering_invariants = ['no numpy import']

    def test_creates_valid_plan(self):
        plan = _build_experiment_plan(
            iteration=5,
            island_epoch_id='island_00_epoch_00',
            direction='try different basis functions',
            decision=self.FakeDecision(),
            baseline={'id': 'rec-best'},
            task=self.FakeTask(),
            candidates_per_context=4,
        )
        assert isinstance(plan, ExperimentPlan)
        plan.validate()
        assert plan.action == 'continue'
        assert plan.target_island_epoch_id == 'island_00_epoch_00'
        assert plan.parent_ids == ['rec-best']
        assert plan.budget['candidate_count'] == 4
        assert plan.budget['sandbox_seconds_per_cell'] == 300
        assert 'no numpy import' in plan.negative_constraints

    def test_plan_id_is_deterministic(self):
        args = dict(
            iteration=5,
            island_epoch_id='island_00_epoch_00',
            direction='try X',
            decision=self.FakeDecision(),
            baseline={'id': 'rec-best'},
            task=self.FakeTask(),
            candidates_per_context=4,
        )
        plan1 = _build_experiment_plan(**args)
        plan2 = _build_experiment_plan(**args)
        assert plan1.id == plan2.id

    def test_different_iterations_different_ids(self):
        args = dict(
            island_epoch_id='island_00_epoch_00',
            direction='try X',
            decision=self.FakeDecision(),
            baseline={'id': 'rec-best'},
            task=self.FakeTask(),
            candidates_per_context=4,
        )
        plan1 = _build_experiment_plan(iteration=1, **args)
        plan2 = _build_experiment_plan(iteration=2, **args)
        assert plan1.id != plan2.id


class TestV5ContextE2E:
    def test_seed_produces_nonempty_context(self, tmp_path):
        bridge = V5Bridge(tmp_path / 'run', num_islands=2)
        bridge.record_seed('seed-0', score=0.5, metrics={'artifact_sha256': 'abc'})
        bridge.record_seed('seed-1', score=0.4, metrics={'artifact_sha256': 'def'})
        epochs = bridge.initialize(['seed-0', 'seed-1'], frozen_baseline_score=0.5)
        epoch_id = f'{epochs[0].island_id}_epoch_{epochs[0].epoch:02d}'
        v5_context = bridge.build_context(epoch_id)
        assert v5_context['portfolio_text']
        assert v5_context['portfolio'] is not None

    def test_plan_event_persisted(self, tmp_path):
        bridge = V5Bridge(tmp_path / 'run', num_islands=2)
        bridge.record_seed('seed-0', score=0.5, metrics={})
        bridge.record_seed('seed-1', score=0.4, metrics={})
        bridge.initialize(['seed-0', 'seed-1'], frozen_baseline_score=0.5)
        plan = ExperimentPlan(
            id='plan_test_001',
            action='continue',
            target_island_epoch_id='island_00_epoch_00',
            generation_operator='local_mutation',
            parent_ids=['seed-0'],
            inspiration_ids=[],
            analogy_hypothesis_id=None,
            implementation_intent='test direction',
            negative_constraints=[],
            success_criterion='improve score',
            budget={
                'candidate_count': 2,
                'sandbox_seconds_per_cell': 300,
                'max_artifact_bytes': 1_000_000,
            },
        )
        plan.validate()
        bridge.event_store.append_plan_event(plan)
        plans = bridge.event_store.read_plan_events()
        assert len(plans) == 1
        assert plans[0].id == 'plan_test_001'

    def test_experiment_plan_id_flows_to_event(self, tmp_path):
        bridge = V5Bridge(tmp_path / 'run', num_islands=2)
        bridge.record_seed('seed-0', score=0.5, metrics={})
        bridge.record_seed('seed-1', score=0.4, metrics={})
        epochs = bridge.initialize(['seed-0', 'seed-1'], frozen_baseline_score=0.5)
        epoch_id = f'{epochs[0].island_id}_epoch_{epochs[0].epoch:02d}'
        bridge.on_candidate_evaluated(
            record_id='cand-0',
            island_epoch_id=epoch_id,
            score=0.6,
            status='ok',
            parent_ids=['seed-0'],
            metrics={'experiment_plan_id': 'plan_test_001'},
        )
        events = bridge.event_store.read_experiment_events()
        cand_events = [e for e in events if e.record_id == 'cand-0']
        assert len(cand_events) == 1
        assert cand_events[0].experiment_plan_id == 'plan_test_001'

    def test_proposal_packet_uses_candidate_seed_and_parent_source(self, tmp_path):
        bridge = V5Bridge(tmp_path / 'run', num_islands=2)
        bridge.record_seed('seed-0', score=0.5, metrics={})
        epochs = bridge.initialize(['seed-0'], base_proposal_seed=42)
        epoch_id = f'{epochs[0].island_id}_epoch_{epochs[0].epoch:02d}'
        plan = ExperimentPlan(
            id='plan_test_002',
            action='continue',
            target_island_epoch_id=epoch_id,
            generation_operator='local_mutation',
            parent_ids=['seed-0'],
            inspiration_ids=[],
            analogy_hypothesis_id=None,
            implementation_intent='test direction',
            negative_constraints=[],
            success_criterion='improve score',
            budget={
                'candidate_count': 2,
                'sandbox_seconds_per_cell': 300,
                'max_artifact_bytes': 1_000_000,
            },
        )
        packet = bridge.build_proposal_context(
            plan, '# FILE: solver.py\nreturn 1\n', candidate_seed=7,
        )['proposal']
        assert packet.parent_source.startswith('# FILE: solver.py')
        assert packet.candidate_seed == 7

    def test_context_packet_is_forwarded_to_context_agent(self, tmp_path, monkeypatch):
        source = tmp_path / 'source'
        source.mkdir()
        (source / 'solver.py').write_text('print(1)\n')
        eb = ExperienceBank(tmp_path / 'eb', direction='max')
        eb.commit(source, 1.0, 'ok', 'seed', None, '')
        task = SimpleNamespace(
            direction='max', metric='score', description='task',
            editable_files=['solver.py'], fallback_directions=['try'],
            engineering_invariants=[], allowed_context_phases=None,
        )
        decision = ContextDecision(
            action='continue', analysis='a', reason='r', expected_gain=0.0,
            confidence=0.5, next_experiment='try',
        )
        captured = {}

        def fake_analysis(*args, **kwargs):
            captured['v5_context_text'] = kwargs.get('v5_context_text')
            return decision

        monkeypatch.setattr('context_agent._llm_context_analysis', fake_analysis)
        build_inspiration(
            task, eb, 0, backend='codex',
            v5_context_prompt='## V5 retrieved context\n\npacket',
        )
        assert 'V5 retrieved context' in captured['v5_context_text']


class TestReconcileAutoResolveRequiresIntegrity:
    def test_orphan_event_blocks_auto_resolve(self, tmp_path):
        bridge = V5Bridge(tmp_path / 'run', num_islands=2)
        bridge.record_seed('seed-0', score=0.5, metrics={})
        bridge.initialize(['seed-0', 'seed-1'])
        bridge.event_store.append_experiment_event(
            ExperimentEvent(
                record_id='rec-0', algorithm_bundle_sha256='',
                experiment_plan_id='plan', island_epoch_id='island_99_epoch_00',
                status='ok', score=0.5, score_metric='score',
                per_instance_metrics_ref='', behavior_profile_ref='',
                runtime_metrics_ref='', parent_ids=[], inspiration_ids=[],
                created_at='now',
            )
        )
        bridge._log_sync_error('rec-0', 'test_op', ValueError('x'))
        result = bridge._reconcile(legacy_record_ids=['seed-0', 'rec-0'])
        assert 'rec-0' in result['orphan_events']
        assert bridge.sync_status == 'degraded'

    def test_missing_card_blocks_auto_resolve(self, tmp_path):
        bridge = V5Bridge(tmp_path / 'run', num_islands=2)
        bridge.record_seed('seed-0', score=0.5, metrics={})
        epochs = bridge.initialize(['seed-0', 'seed-1'])
        epoch_id = f'{epochs[0].island_id}_epoch_{epochs[0].epoch:02d}'
        bridge.on_candidate_evaluated(
            record_id='rec-0', island_epoch_id=epoch_id,
            score=0.5, status='ok', parent_ids=[], metrics={},
        )
        bridge._log_sync_error('rec-0', 'test_op', ValueError('x'))
        bridge._cards.pop('rec-0', None)
        bridge._reconcile(legacy_record_ids=['seed-0', 'rec-0'])
        assert bridge.sync_status == 'degraded'

    def test_new_error_after_resolution_stays_degraded(self, tmp_path):
        bridge = V5Bridge(tmp_path / 'run', num_islands=2)
        bridge._log_sync_error('rec-0', 'first', ValueError('x'))
        bridge.resolve_sync_error('rec-0', 'fixed')
        bridge._log_sync_error('rec-0', 'second', ValueError('y'))
        assert bridge.sync_status == 'degraded'


class TestReviewStatusFiltering:
    def test_failed_scored_event_is_not_review_evidence(self, tmp_path, monkeypatch):
        bridge = V5Bridge(tmp_path / 'run', num_islands=2)
        epochs = bridge.initialize(['seed-0'])
        ids = [f'{epoch.island_id}_epoch_{epoch.epoch:02d}' for epoch in epochs]
        bridge.on_candidate_evaluated(
            'failed', ids[0], 100.0, 'crash', [], {},
        )
        bridge.on_candidate_evaluated(
            'ok-0', ids[0], 0.5, 'ok', [], {},
        )
        bridge.on_candidate_evaluated(
            'ok-1', ids[1], 1.0, 'ok', [], {},
        )
        captured = {}

        def fake_review(round_number, scores):
            captured['scores'] = scores
            return {}

        monkeypatch.setattr(bridge.island_scheduler, 'run_review', fake_review)
        bridge.on_context_complete(10)
        assert 'failed' not in captured['scores']
        assert captured['scores']['ok-1'] == 1.0
