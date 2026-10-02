import json
import hashlib
import re
from pathlib import Path
from types import SimpleNamespace
import pytest

from harness import (
    _build_v5_prompt_section,
    _inject_v5_context,
    _inject_v5_proposal,
    _build_experiment_plan,
    run_pipeline,
)
from harness_v5 import V5Bridge
from schemas_v5 import ExperimentEvent, ExperimentPlan
from eb import ExperienceBank
from context_agent import ContextDecision, build_inspiration
from proposal_agent import prepare_draft
from reporting import export_bundle
from sandbox import source_tree_hash


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


class TestV5PipelineEndToEnd:
    def test_pipeline_carries_context_plan_proposal_and_event(self, tmp_path, monkeypatch):
        source = tmp_path / 'source'
        source.mkdir()
        (source / 'solver.py').write_text("print('seed')\n")
        eb = ExperienceBank(tmp_path / 'eb', direction='max')
        seed = eb.commit(source, 1.0, 'ok', 'seed', None, '')
        bridge = V5Bridge(tmp_path / 'run', num_islands=2)
        bridge.record_seed(seed['id'], seed['score'], seed['metrics'])
        bridge.initialize([seed['id']], base_proposal_seed=100)

        task = SimpleNamespace(
            run_dir=tmp_path / 'run', eval_concurrency=1,
            candidates_per_context=2, candidate_repair_attempts=0,
            research_revision_attempts=0, editable_files=['solver.py'],
            direction='max', protocol='test-v1', run_id='v5-e2e',
            timeout_s=10, max_artifact_bytes=100_000, max_output_mb=1,
            description='A deterministic V5 pipeline test.', metric='score',
            fallback_directions=['try a bounded mutation'],
            engineering_invariants=[], allowed_context_phases=None,
            candidate_instructions='',
        )

        context_prompts = []
        proposal_prompts = []

        class AgentResult:
            returncode = 0
            stderr = ''
            stdout = json.dumps({
                'action': 'continue',
                'analysis': 'Use the retrieved island evidence.',
                'reason': 'A focused next experiment remains useful.',
                'expected_gain': 0.1,
                'confidence': 0.8,
                'phase': 'numeric',
                'target_claim_id': None,
                'success_criterion': 'score improves',
                'next': 'try a bounded mutation',
            })

        def fake_context_agent(prompt, **_kwargs):
            context_prompts.append(prompt)
            return AgentResult()

        def fake_proposal_agent(parent_dir, draft_dir, prompt, editable_files, **_kwargs):
            proposal_prompts.append(prompt)
            prepare_draft(parent_dir, draft_dir)
            (draft_dir / editable_files[0]).write_text("print('candidate')\n")
            return True, 'bounded deterministic proposal'

        def fake_evaluator(_solution_dir, sandbox_dir, _task):
            sandbox_dir = Path(sandbox_dir)
            sandbox_dir.mkdir(parents=True, exist_ok=True)
            (sandbox_dir / 'run.log').write_text('deterministic evaluator\n')
            return 1.1, 'ok', 'deterministic evaluator', {
                'set_hash': 'deterministic-set',
            }

        monkeypatch.setattr('context_agent.run_agent', fake_context_agent)
        monkeypatch.setattr('harness.propose', fake_proposal_agent)
        monkeypatch.setattr('harness.run_solution', fake_evaluator)

        outcome = run_pipeline(
            task, eb, iterations=1, workers=2, backend='codex',
            model='offline-test', trial_seed=0, v5_bridge=bridge,
        )

        assert outcome['reason'] == 'iteration_limit'
        assert len(context_prompts) == 1
        assert '## V5 retrieved context' in context_prompts[0]
        assert '## V5 Portfolio Context' in context_prompts[0]
        assert len(proposal_prompts) == 2
        assert all('## V5 Proposal Context' in prompt for prompt in proposal_prompts)
        assert all('# FILE: solver.py' in prompt for prompt in proposal_prompts)
        packet_seeds = {
            re.search(r'## Candidate Seed\n(\d+)', prompt).group(1)
            for prompt in proposal_prompts
        }
        assert packet_seeds == {'0', '1'}

        plans = bridge.event_store.read_plan_events()
        events = bridge.event_store.read_experiment_events()
        assert len(plans) == 1
        assert len(events) == 3
        candidate_events = [event for event in events if event.record_id != seed['id']]
        assert {event.experiment_plan_id for event in candidate_events} == {plans[0].id}
        assert all(event.status == 'ok' for event in candidate_events)
        records = eb.records()
        assert all(record['metadata']['v5_status'] == 'ready' for record in records[1:])
        assert bridge.sync_status == 'healthy'

        resumed = V5Bridge(task.run_dir, num_islands=2)
        resumed.initialize([seed['id']], base_proposal_seed=100)
        assert resumed.sync_status == 'healthy'
        assert len(resumed.event_store.read_plan_events()) == 1
        assert len(resumed.event_store.read_experiment_events()) == 3
        assert resumed.get_island_diagnostics()['cards_cached'] == 3


class TestV5BundleExport:
    def test_bundle_contains_v5_event_tree_and_counts(self, tmp_path):
        source = tmp_path / 'source'
        source.mkdir()
        (source / 'solver.py').write_text("print('seed')\n")
        eb = ExperienceBank(tmp_path / 'eb', direction='max')
        solver_hash = hashlib.sha256((source / 'solver.py').read_bytes()).hexdigest()
        source_hash = source_tree_hash(source, 1_000_000)[0]
        eb.commit(
            source, 1.0, 'ok', 'seed', None, '',
            metrics={'source_snapshot_sha256': source_hash},
            metadata={'editable_file_sha256': {'solver.py': solver_hash}},
        )

        v5_events = tmp_path / 'run' / 'v5' / 'events' / 'events'
        v5_events.mkdir(parents=True)
        (v5_events / 'experiment_events.jsonl').write_text('{"record_id":"seed"}\n')
        (v5_events / 'plan_events.jsonl').write_text('{"id":"plan-0"}\n')
        (v5_events / 'annotation_events.jsonl').write_text('')
        (v5_events / 'analogy_results.jsonl').write_text('')
        (tmp_path / 'run' / 'v5' / 'islands.json').write_text('{"epochs": []}\n')

        task = SimpleNamespace(
            name='test', protocol='test-v1', run_id='bundle-v5',
            editable_files=['solver.py'], run_dir=tmp_path / 'run',
        )
        destination = tmp_path / 'bundle'
        export_bundle(
            task, eb, destination, root=tmp_path,
            run_manifest={'manifest_sha256': 'bundle-manifest'},
        )

        assert (destination / 'v5' / 'islands.json').is_file()
        assert (destination / 'v5' / 'events' / 'events' / 'experiment_events.jsonl').is_file()
        manifest = json.loads((destination / 'manifest.json').read_text())
        assert manifest['v5']['present'] is True
        assert manifest['v5']['experiment_event_count'] == 1
        assert manifest['v5']['plan_event_count'] == 1
        assert manifest['v5']['file_count'] >= 3
