"""One research loop with controlled information views and durable ex ante records."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from stochopia.common import atomic_write_json
from .code_dedup import find_duplicate
from .folding_ledger import FoldingLedger, promotion_decision
from .prior_cards import PriorCard, shuffled_cards
from .proposal import Proposal, StructuredPrediction, parse_proposal
from .call_budget import BudgetExceededError, CallBudgetGuard
from .provider import ProviderAttemptError
from .contracts import Messages, Transport

Json = dict[str, Any]
Arm = Literal['no_llm', 'greedy_score_llm', 'residual_d_llm', 'full_zd_llm',
              'd_only', 'true_z_d', 'shuffled_z_d']
PROMPT_CAP = 4000
SOURCE_TOKEN_CAP = 1050
# Exact estimator definitions and null benchmarks only; no empirical thresholds.
# Finite-window nulls matter: with n ~ 100 trades per session, independent signs
# give sign_acf_sum ~ -0.33 and flow_vr ~ 0.5, not the population values 0 and 1.
Z_PREAMBLE = 'Per 256-event training session (n ~ 100 aggressive trades), then median and IQR across sessions.'
Z_REFERENCE = {
    'sign_acf_sum_1_32': ('sum of trade-sign autocorrelations over event lags 1..32; 0 for independent signs, '
                          'finite-window null about -32/n; higher = multi-lag directional persistence'),
    'flow_vr_32_4': ('per-trade variance ratio of 32- vs 4-trade signed-flow sums; 1.0 for uncorrelated signs, '
                     'finite-window null about 0.5; higher = multi-step persistence, lower = mean reversion'),
    'trade_to_top1_depth': '>=0; median trade size over consumed best-quote depth (1 = a median trade clears the best queue)',
    'midprice_shift_rate': 'in [0, 1]; fraction of quoted events that move the midprice (0 = stationary best quotes; not spread width)',
    'cancel_to_limit_ratio': '>=0; volume canceled in the best three levels over all newly posted limit volume',
    'zero_wait_prob': 'in [0, 1]; fraction of events with zero recorded wait (depends on clock resolution and arrival rate)',
    'interarrival_cv2': 'var/mean^2 of waits; 1.0 for stationary Poisson arrivals; >1 clustering or rate nonstationarity',
    'book_imbalance_bias': ('in [0, 1]; |session mean| of ten-level (bid-ask)/(bid+ask) depth; 0 in expectation under '
                            'symmetric depth, >0 in finite windows'),
    'flow_sign_bias': ('in [0, 1]; |session mean| of buy(+1)/sell(-1) signs; 0 in expectation under balanced flow, '
                       'about sqrt(2/(pi*n)) for independent signs'),
}
Z_KEYS = tuple(Z_REFERENCE)
D_GROUPS = (
    ('nll_overview', ('val_nll_total', 'event_nll', 'level_nll', 'size_nll')),
    ('timing_clustering', ('dt_nll', 'zero_wait_rate_error', 'interarrival_cv2_rel_error')),
    ('rollout_viability', ('exhausted_replicate_rate', 'available_quote_fraction')),
    ('flow_memory', ('sign_acf_abs_error', 'memory_gap_64')),
    ('impact_response', ('counterfactual_impact_mae', 'post_stop_relaxation_delta')),
    ('drift_stability', ('free_abs_drift_ticks', 'c2_bootstrap_pass_rate')),
)
D_KEYS = tuple(key for _, keys in D_GROUPS for key in keys)
GREEDY_KEYS = ('val_nll_total', 'counterfactual_impact_mae')
CF = 'counterfactual_impact_mae'
OMITTED = 'Regime statistics omitted in this control condition; rely on model residuals and structural boundaries.'

SEED_SOURCE = '''from torch import nn
from stochopia.neural.probe_features import ProbeBase, input_projection

class CandidateLOBWorldModel(ProbeBase):
    def __init__(self):
        super().__init__()
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, 60, batch_first=True)

    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        return self.gru(x)[0]

    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        prior = x.new_zeros((x.size(0), 60)) if state is None else state
        _, h = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        return h[0], h[0]
'''

SYSTEM = '''You are studying a causal limit-order-book world model. Diagnose structure in the market data profile and model residuals, then propose one substantive tensor operator change to the current parent and register a falsifiable prediction BEFORE seeing results. The objective is accurate response to a fixed external order, retaining likelihood and valid quotes. Use only causal StepObservation, not future outcomes. CandidateLOBWorldModel must subclass LOBWorldModel, implement sequence_context, step_context and head_logits (or inherit ProbeBase); legal four heads, finite gradients, <=28000 trainable parameters.
Exact tensor contract:
- step_context(self, obs: StepObservation, state): obs.book_ticks and obs.book_shares each [B, 2, 10]; side axis 0=bid, 1=ask, level axis 0..9 (20 elements per sample, NEVER reshape/view to just 10). obs.previous_event [B, 4]; obs.is_own [B]. Return (context, next_state), context [B, 60].
- sequence_context(self, obs: StepObservation): obs.book_ticks and obs.book_shares each [B, T, 2, 10], obs.previous_event [B, T, 4], obs.is_own [B, T]. Return [B, T, 60], numerically equivalent to successive step_context calls.
- Recurrent state may be [B, 60] or tensors/tuples/dicts totaling <=60 real scalars per sample, e.g. [B, 10, 6] or ([B, 40], [B, 2, 10]). ProbeBase.raw_features returns 75 features; inherited heads consume 60 context features.
Available stochopia.neural.probe_features helpers: input_projection, level_inputs, advect_level_state, previous_flow, volume_clock; torch and standard mathematical primitives permitted. Keep complete source below 1050 tokens with explicit changed computation. Use torch.int64 for integer dtype. No file/network operations. Do not merely tune constants or choose a prewritten probe. No extra retries.'''

OBJECTIVE = '''[5. Objective & Output Schema]
Deltas: candidate minus parent on paired days. Guardrails always: dNLL<=+0.005, dExhaust<=+0.05.
Response measured on >=6 common valid days: promote if dResponse<=-0.010 tick, or |dResponse|<0.010 with dExhaust<=-0.10 or dNLL<=-0.008 (objective_improvement).
Otherwise: promote if dExhaust<=-0.10 or dNLL<=-0.008 (viability_recovery). NA means unavailable, never zero.
Reply with one JSON fenced block, then one complete Python fenced block. JSON keys in order:
data_regime_diagnosis: structure in the data profile or residuals worth exploiting, with evidence;
selected_card_ids: related catalog IDs, primary first, at most three; [] for a structure outside the catalog;
regime_hypothesis: the mechanism, and the data conditions where it should be ineffective or harmful;
target_metric: dt_nll | exhausted_replicate_rate | counterfactual_impact_mae | sign_acf_abs_error;
expected_direction: decrease | increase;
unfold_ids: optional, up to two ledger IDs to expand.'''


@dataclass(frozen=True)
class View:
    arm: Arm
    env_id: str
    z: Mapping[str, Any] | None

    @property
    def greedy(self) -> bool:
        return self.arm == 'greedy_score_llm'

    @property
    def use_cards(self) -> bool:
        return not self.greedy and self.arm != 'no_llm'


def source_hash(source: str) -> str:
    return sha256(source.encode()).hexdigest()


def _level_text(value: object) -> str:
    return format(value, '.6g') if type(value) in (int, float) else 'NA'


def _z_text(z: Mapping[str, Any] | None) -> str:
    if z is None:
        return OMITTED
    lines = [Z_PREAMBLE, 'metric | median | IQR | reference_and_interpretation']
    for key, reference in Z_REFERENCE.items():
        summary = z[key]
        lines.append(f'{key} | {_level_text(summary["median"])} | {_level_text(summary["intraday_iqr"])} | {reference}')
    return '\n'.join(lines)


def _d_text(view: View, result: Json) -> str:
    levels = result['aggregate']
    if view.greedy:
        return '\n'.join(f'{key}: {_level_text(levels.get(key))}' for key in GREEDY_KEYS)
    support = result.get('support', {}).get(CF, {})
    lines = []
    for group, keys in D_GROUPS:
        values = []
        for key in keys:
            value = f'{key}={_level_text(levels.get(key))}'
            if key == CF and 'paired_days' in support:
                value += f' ({support["paired_days"]}/{support["registered_days"]} valid days)'
            values.append(value)
        lines.append(f'{group}: ' + ', '.join(values))
    return '\n'.join(lines)


def _sections(view: View, parent_id: str, source: str, result: Json,
              cards: Sequence[PriorCard]) -> tuple[str, str]:
    """Data first; the three isomorphic arms differ only inside section 1."""
    cards_text = '\n\n'.join(card.render() for card in cards) or 'None.'
    head = (f'Anonymous environment: {view.env_id}\n'
            f'[1. Market Data Profile Z]\n{_z_text(view.z)}\n\n'
            f'[2. Prior Mechanism Catalog (C01-C10, Open Reference)]\n{cards_text}\n\n'
            f'[3. Core Parent Residuals D & Trial Ledger]\n{_d_text(view, result)}\n')
    tail = (f'\n\n[4. Current Parent Source Code]\n{parent_id} | params={result["params"]} state={result["state"]}\n'
            f'```python\n{source.rstrip()}\n```\n\n{OBJECTIVE}')
    return head, tail


def _greedy_history(ledger: FoldingLedger) -> str:
    lines = []
    for row in ledger.records:
        delta = row.get('deltas', {})
        def estimate(key: str) -> object:
            item = delta.get(key)
            return item.get('estimate') if isinstance(item, dict) else item
        lines.append(f'{row["cand_id"]} {row["outcome"]}: '
                     f'dNLL={_level_text(estimate("val_nll_total"))}, '
                     f'dCF={_level_text(estimate("counterfactual_impact_mae"))}')
    return '\n'.join(lines)


def build_messages(view: View, parent_id: str, source: str, result: Json,
                   cards: Sequence[PriorCard], ledger: FoldingLedger,
                   unfold_ids: Sequence[str], count: Callable[[Messages], int]) -> tuple[Messages, int]:
    """Show the complete parent, D, Z and catalog; fold only the optional ledger history."""
    head, tail = _sections(view, parent_id, source, result, cards)
    def packets(history: str) -> Messages:
        return [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': head + '\nTrial ledger:\n' + history + tail}]
    minimum = count(packets(''))
    if minimum > PROMPT_CAP:
        raise ValueError('complete parent and required context exceed prompt budget')
    if view.greedy:
        history = _greedy_history(ledger)
        if count(packets(history)) > PROMPT_CAP:
            history = ''
    else:
        # The ledger only offers rows whose in-context token cost still fits.
        history = ledger.render(parent_id, tuple(unfold_ids), max_tokens=PROMPT_CAP - minimum,
                                token_count=lambda text: count(packets(text)) - minimum)
    messages = packets(history)
    return messages, count(messages)


def prediction_verdict(prediction: Json, paired: Json) -> str:
    """Directional confidence requires the complete 95% interval on one side of zero."""
    metric = paired.get(prediction['target_metric'], {})
    interval = metric.get('interval')
    if interval is None:
        return 'INCONCLUSIVE'
    lo, hi = interval
    if prediction['expected_direction'] == 'decrease':
        return 'SUPPORTED' if hi < 0 else 'REFUTED' if lo > 0 else 'INCONCLUSIVE'
    return 'SUPPORTED' if lo > 0 else 'REFUTED' if hi < 0 else 'INCONCLUSIVE'


def deterministic_proposal(round_number: int) -> Proposal:
    names = ('gru-multiscale', 'gru-surprise', 'matrix-noshift', 'matrix-advect',
             'flat-readoutgate', 'selective-ssm')
    name = names[(round_number - 1) % len(names)]
    # Explicit constructor keeps a deterministic, actual variant instead of a class alias.
    source = ('from stochopia.neural.probes import make_probe\n'
              'from stochopia.neural.lob_interface import LOBWorldModel\n'
              'class CandidateLOBWorldModel(LOBWorldModel):\n'
              '    def __new__(cls):\n'
              f'        return make_probe({name!r})\n')
    metric = 'exhausted_replicate_rate' if name.startswith('matrix') else 'counterfactual_impact_mae'
    pred = StructuredPrediction(data_regime_diagnosis='Fixed probe sequence.', selected_card_ids=(),
                                regime_hypothesis='Fixed probe sequence.', target_metric=metric,
                                expected_direction='decrease')
    return Proposal(source=source, prediction=pred)


# --- Round execution; no evaluator receives prompt or Z ---

class DiscoveryLoop:
    def __init__(self, *, view: View, root: Path, seed_result: Json,
                 evaluator: Callable[[str], Json], pair: Callable[[Json, Json], Json],
                 count: Callable[[Messages], int], transport: Transport | None = None,
                 guard: CallBudgetGuard | None = None, rounds: int = 6,
                 should_stop: Callable[[], bool] = lambda: False, call_prefix: str | None = None,
                 replicate_id: str = 'rep_0') -> None:
        self.view, self.root, self.seed_result = view, root, seed_result
        self.evaluator, self.pair, self.count = evaluator, pair, count
        self.transport, self.guard, self.rounds = transport, guard, rounds
        self.should_stop, self.replicate_id = should_stop, replicate_id
        self.call_prefix = call_prefix if call_prefix is not None else f'{view.env_id}/{view.arm}'
        self.ledger = FoldingLedger(root)
        if not self.ledger.records:
            self.ledger.append({'cand_id': 'B0', 'parent_id': '', 'outcome': 'SEED',
                                'source': SEED_SOURCE, 'levels': seed_result['aggregate'],
                                'parameters': seed_result['params'], 'state_scalars': seed_result['state'],
                                'source_sha256': source_hash(SEED_SOURCE), 'evaluation': seed_result})

    def cards(self, round_number: int) -> list[PriorCard]:
        return shuffled_cards(self.view.env_id, self.replicate_id, round_number) if self.view.use_cards else []

    def run(self) -> Json:
        parent_id, parent_source, parent = 'B0', SEED_SOURCE, self.seed_result
        unfold: Sequence[str] = ()
        # Reconstruct only already completed rounds; no replayed requests.
        for row in self.ledger.records[1:]:
            unfold = tuple(row.get('unfold_ids', ()))
            if row['outcome'] == 'PROMOTED':
                parent_id, parent_source, parent = row['cand_id'], row['source'], row['evaluation']
        completed = len(self.ledger.records) - 1
        for number in range(completed + 1, self.rounds + 1):
            if self.should_stop():
                raise RuntimeError('study interrupted before another proposal')
            cid = f'c{number:02d}'
            cards = self.cards(number)
            messages, tokens = build_messages(self.view, parent_id, parent_source, parent,
                                               cards, self.ledger, unfold, self.count)
            order = [item.card_id for item in cards]
            record: Json = {'round': number, 'cand_id': cid, 'parent_id': parent_id,
                            'parent_source': parent_source, 'prompt': messages, 'prompt_tokens': tokens,
                            'card_order': order, 'outcome': 'INVALID_PROPOSAL', 'promotion_reason': None,
                            'source': '', 'raw_reply': '', 'prediction': {}, 'deltas': {}, 'levels': {},
                            'selected_card_ids': [], 'unfold_ids': [], 'parameters': None, 'state_scalars': None}
            directory = self.root / 'preregistered'
            directory.mkdir(parents=True, exist_ok=True)
            atomic_write_json(directory / f'{cid}-request.json', {'messages': messages, 'tokens': tokens,
                              'parent_id': parent_id, 'card_order': order, 'parent_sha256': source_hash(parent_source)})
            try:
                if self.view.arm == 'no_llm':
                    proposal = deterministic_proposal(number)
                else:
                    if self.transport is None or self.guard is None:
                        raise RuntimeError('language-model arm requires a metered transport')
                    call_id = f'{self.call_prefix}/{cid}'
                    reply = self.guard.result(call_id, messages=messages)
                    if reply is None:
                        reply = self.guard.call(self.transport, messages, call_id=call_id)
                    record['raw_reply'] = reply.text
                    record['usage'] = {'input_tokens': reply.input_tokens, 'output_tokens': reply.output_tokens}
                    if reply.error:
                        raise ValueError(reply.error)
                    proposal = parse_proposal(reply.text)
                record['source'] = proposal.source
                prediction = asdict(proposal.prediction)
                record['prediction'] = prediction
                record['selected_card_ids'] = list(prediction['selected_card_ids'])
                unfold_next = tuple(prediction.get('unfold_ids', ()))
                if any(key not in {row['cand_id'] for row in self.ledger.records} for key in unfold_next):
                    raise ValueError('unknown unfold node')
                record['unfold_ids'] = list(unfold_next)
                atomic_write_json(directory / f'{cid}-prediction.json', {'prediction': prediction,
                                  'source_sha256': source_hash(proposal.source)})
                # A single source-size ceiling applies to every information arm, so the
                # next prompt can display the complete parent, never a cropped source.
                source_size = self.count([{'role': 'user', 'content': proposal.source}])
                source_size -= self.count([{'role': 'user', 'content': ''}])
                if source_size > SOURCE_TOKEN_CAP:
                    record['outcome'] = 'PROMPT_CAPACITY_REJECTED'
                else:
                    duplicate = find_duplicate(proposal.source, self.ledger.evaluated_sources())
                    if duplicate is not None:
                        record['outcome'] = 'DUPLICATE_REJECTED'
                        record['duplicate'] = asdict(duplicate)
                    else:
                        candidate = self.evaluator(proposal.source)
                        record['evaluation'] = candidate
                        record['source_sha256'] = source_hash(proposal.source)
                        record['parameters'] = candidate.get('params')
                        record['state_scalars'] = candidate.get('state')
                        if candidate['status'] != 'VALID':
                            record['outcome'] = candidate['status']
                            record['error'] = candidate.get('error')
                        else:
                            paired = self.pair(candidate, parent)
                            record['levels'] = candidate['aggregate']
                            record['deltas'] = paired
                            record['prediction_verdict'] = prediction_verdict(prediction, paired)
                            record['outcome'], record['promotion_reason'] = promotion_decision(
                                parent['aggregate'], candidate['aggregate'], paired)
            except BudgetExceededError as exc:
                record['outcome'] = 'INTERRUPTED_CALL'
                record['error'] = str(exc)
            except ProviderAttemptError as exc:
                record['outcome'] = 'PROVIDER_ERROR'
                record['raw_reply'] = exc.raw_reply or ''
                record['error'] = str(exc)
                record['usage'] = {'input_tokens': exc.input_tokens, 'output_tokens': exc.output_tokens}
                if exc.status_code in (401, 403):
                    raise RuntimeError('provider authentication failed during the registered study') from exc
            except ValueError as exc:
                record['error'] = str(exc)
            finally:
                self.ledger.append(record)
            print(f'ROUND {self.call_prefix}/{cid} {record["outcome"]}', flush=True)
            if record['outcome'] == 'PROMOTED':
                parent_id, parent_source, parent = cid, record['source'], record['evaluation']
            unfold = tuple(record['unfold_ids'])
        return {'final_id': parent_id, 'final_sha256': source_hash(parent_source), 'rounds': self.rounds,
                'records': self.ledger.records}
