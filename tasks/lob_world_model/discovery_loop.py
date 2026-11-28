"""One research loop with controlled information views and durable ex ante records."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Literal

from stochopia.common import atomic_write_json
from .code_dedup import find_duplicate
from .folding_ledger import FoldingLedger, promotion_decision
from .prior_cards import PriorCard, advance_cooldown, select_active_cards
from .proposal import Proposal, StructuredPrediction, parse_proposal
from .call_budget import BudgetExceededError, CallBudgetGuard
from .provider import ProviderAttemptError
from .contracts import Messages, Transport

Json = dict[str, Any]
Arm = Literal['no_llm', 'greedy_score_llm', 'residual_d_llm', 'full_zd_llm',
              'd_only', 'true_z_d', 'shuffled_z_d']
PROMPT_CAP = 2800
SOURCE_TOKEN_CAP = 1050
Z_KEYS = ('sign_acf_sum_1_32', 'flow_vr_32_4', 'trade_to_top1_depth',
          'midprice_shift_rate', 'cancel_to_limit_ratio', 'zero_wait_prob',
          'interarrival_cv2', 'book_imbalance_bias', 'flow_sign_bias')
D_KEYS = ('val_nll_total', 'dt_nll', 'event_nll', 'level_nll', 'size_nll',
          'counterfactual_impact_mae', 'stylized_impact_mae', 'post_stop_relaxation_delta',
          'visible_minus_blind_auc', 'c2_bootstrap_pass_rate', 'exhausted_replicate_rate',
          'sign_acf_abs_error', 'memory_gap_1', 'memory_gap_4', 'memory_gap_16',
          'memory_gap_64', 'zero_wait_rate_error', 'interarrival_cv2_error', 'interarrival_cv2_rel_error',
          'free_abs_drift_ticks', 'available_quote_fraction')
GREEDY_KEYS = ('val_nll_total', 'counterfactual_impact_mae')
TARGETS = ('dt_nll', 'exhausted_replicate_rate', 'counterfactual_impact_mae', 'sign_acf_abs_error')
# Match environment labels, not substrings of ordinary words (e.g. within).
LABELS = re.compile(r'(?i)(?<![A-Za-z0-9])(?:w[26]|thin|thick|hawkes|long)(?![A-Za-z0-9])')
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

SYSTEM = '''You are studying a causal limit-order-book world model. Propose one substantive tensor operator change to the current parent and make a falsifiable metric prediction BEFORE seeing results. The objective is accurate response to a fixed external order, retaining likelihood and valid quotes. Use only causal StepObservation, not labels or future outcomes. CandidateLOBWorldModel must subclass LOBWorldModel, implement sequence_context, step_context and head_logits (or inherit ProbeBase); legal four heads, finite gradients, <=28000 trainable parameters.
Exact tensor contract:
- step_context(self, obs: StepObservation, state): obs.book_ticks and obs.book_shares each [B, 2, 10]; side axis 0=bid, 1=ask, level axis 0..9 (20 elements per sample, NEVER reshape/view to just 10). obs.previous_event [B, 4]; obs.is_own [B]. Return (context, next_state), context [B, 60].
- sequence_context(self, obs: StepObservation): obs.book_ticks and obs.book_shares each [B, T, 2, 10], obs.previous_event [B, T, 4], obs.is_own [B, T]. Return [B, T, 60], numerically equivalent to successive step_context calls.
- Recurrent state may be [B, 60] or tensors/tuples/dicts totaling <=60 real scalars per sample, e.g. [B, 10, 6] or ([B, 40], [B, 2, 10]). ProbeBase.raw_features returns 75 features; inherited heads consume 60 context features.
Available stochopia.neural.probe_features helpers: input_projection, level_inputs, advect_level_state, previous_flow, volume_clock; torch and standard mathematical primitives permitted. Keep complete source below 1050 tokens; prefer explicit changed computation, neutral identifiers and comments; do not label data regimes in source or reply. Use torch.int64 for integer dtype. No file/network operations. A candidate that cannot fit in the next full-parent context is not admissible. Do not merely tune constants or choose a prewritten probe. No extra retries. Return exactly one JSON fenced block followed by one complete Python fenced block. JSON keys: selected_card_ids (list, primary card first, only displayed cards), target_metric (dt_nll, exhausted_replicate_rate, counterfactual_impact_mae, or sign_acf_abs_error), expected_direction (decrease or increase), regime_hypothesis (brief falsifiable explanation), unfold_ids (optional list of at most two existing IDs).'''


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
    lines = ['metric | median | IQR']
    for key in Z_KEYS:
        summary = z[key]
        lines.append(f'{key} | {_level_text(summary["median"])} | {_level_text(summary["intraday_iqr"])}')
    return '\n'.join(lines)


def _base_user(view: View, parent_id: str, source: str, result: Json,
               cards: Sequence[PriorCard]) -> str:
    if view.env_id not in ('env_alpha', 'env_beta'):
        raise ValueError('environment must be anonymous')
    levels = result['aggregate']
    keys = GREEDY_KEYS if view.greedy else D_KEYS
    d = '\n'.join(f'{key}: {_level_text(levels.get(key))}' for key in keys)
    cards_text = '\n\n'.join(card.render() for card in cards) or 'None.'
    # Identical layout for the three isomorphic arms; only the Z block varies.
    return (f'Anonymous environment: {view.env_id}\n'
            f'[Current Parent]\n{parent_id} | params={result["params"]} state={result["state"]}\n'
            f'```python\n{source.rstrip()}\n```\n\n'
            f'[Macro Data Regime Z]\n{_z_text(view.z)}\n\n'
            f'[Model Residuals D]\n{d}\n\n'
            f'[Active Prior Cards]\n{cards_text}\n\n'
            '[Objective]\nBoth guardrails: dNLL<=+0.005, dExhaust<=+0.05. '
            'Improve response error by >=0.010 tick, or when |dResponse|<0.010 '
            'improve exhaust by >=0.10 or NLL by >=0.008. NA is unavailable, never zero.\n\n')


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
    """Preserve full parent and D/cards. Trim optional history, never hidden source text."""
    base = _base_user(view, parent_id, source, result, cards)
    def packets(history: str) -> Messages:
        return [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': base + '[Folded Lineage]\n' + history}]
    minimum = count(packets(''))
    if minimum > PROMPT_CAP:
        raise ValueError('complete parent and required context exceed prompt budget')
    remaining = PROMPT_CAP - minimum
    while True:
        if view.greedy:
            history = _greedy_history(ledger)
            if count(packets(history)) > PROMPT_CAP:
                history = ''
        else:
            history = ledger.render(parent_id, tuple(unfold_ids), max_tokens=remaining,
                                    token_count=lambda text: max(0, count(packets(text)) - minimum))
        messages = packets(history)
        tokens = count(messages)
        if tokens <= PROMPT_CAP:
            break
        if remaining == 0:
            raise ValueError('folded prompt exceeds budget')
        remaining = max(0, remaining - max(16, tokens - PROMPT_CAP))
    if any(LABELS.search(message['content']) for message in messages):
        raise ValueError('anonymous prompt contains a reserved label')
    return messages, tokens


def _prediction_dict(prediction: StructuredPrediction) -> Json:
    return asdict(prediction)


def _estimate(pair: Json, key: str) -> float | None:
    value = pair.get(key, {})
    if isinstance(value, dict):
        return value.get('estimate')
    return value


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


def _adverse(paired: Json) -> bool:
    return any(value is not None and value > threshold for value, threshold in (
        (_estimate(paired, 'val_nll_total'), 0.005),
        (_estimate(paired, 'counterfactual_impact_mae'), 0.01),
        (_estimate(paired, 'exhausted_replicate_rate'), 0.05),
    ))


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
    pred = StructuredPrediction(selected_card_ids=(), target_metric=metric,
                                expected_direction='decrease', regime_hypothesis='Fixed probe sequence.', unfold_ids=())
    return Proposal(source=source, prediction=pred)


# --- Round execution; no evaluator receives prompt or Z ---

class DiscoveryLoop:
    def __init__(self, *, view: View, root: Path, seed_result: Json,
                 evaluator: Callable[[str], Json], pair: Callable[[Json, Json], Json],
                 count: Callable[[Messages], int], transport: Transport | None = None,
                 guard: CallBudgetGuard | None = None, rounds: int = 6,
                 should_stop: Callable[[], bool] = lambda: False, call_prefix: str | None = None) -> None:
        self.view, self.root, self.seed_result = view, root, seed_result
        self.evaluator, self.pair, self.count = evaluator, pair, count
        self.transport, self.guard, self.rounds = transport, guard, rounds
        self.should_stop = should_stop
        self.call_prefix = call_prefix if call_prefix is not None else f'{view.env_id}/{view.arm}'
        self.ledger = FoldingLedger(root)
        if not self.ledger.records:
            self.ledger.append({'cand_id': 'B0', 'parent_id': '', 'outcome': 'SEED',
                                'source': SEED_SOURCE, 'levels': seed_result['aggregate'],
                                'parameters': seed_result['params'], 'state_scalars': seed_result['state'],
                                'source_sha256': source_hash(SEED_SOURCE), 'evaluation': seed_result})

    def run(self) -> Json:
        parent_id, parent_source, parent = 'B0', SEED_SOURCE, self.seed_result
        cooldown: dict[str, int] = {}
        unfold: Sequence[str] = ()
        # Reconstruct only already completed rounds; no replayed requests.
        for row in self.ledger.records[1:]:
            cooldown = dict(row.get('cooldown_after', {}))
            unfold = tuple(row.get('unfold_ids', ()))
            if row['outcome'] == 'PROMOTED':
                parent_id, parent_source, parent = row['cand_id'], row['source'], row['evaluation']
        completed = len(self.ledger.records) - 1
        for number in range(completed + 1, self.rounds + 1):
            if self.should_stop():
                raise RuntimeError('study interrupted before another proposal')
            cid = f'c{number:02d}'
            cards = select_active_cards(parent['aggregate'], cooldown) if self.view.use_cards else []
            messages, tokens = build_messages(self.view, parent_id, parent_source, parent,
                                               cards, self.ledger, unfold, self.count)
            active = [item.card_id for item in cards]
            record: Json = {'round': number, 'cand_id': cid, 'parent_id': parent_id,
                            'parent_source': parent_source, 'prompt': messages, 'prompt_tokens': tokens,
                            'active_card_ids': active, 'cooldown_before': cooldown, 'outcome': 'INVALID_PROPOSAL',
                            'source': '', 'raw_reply': '', 'prediction': {}, 'deltas': {}, 'levels': {},
                            'selected_card_ids': [], 'unfold_ids': [], 'parameters': None, 'state_scalars': None}
            directory = self.root / 'preregistered'
            directory.mkdir(parents=True, exist_ok=True)
            atomic_write_json(directory / f'{cid}-request.json', {'messages': messages, 'tokens': tokens,
                              'parent_id': parent_id, 'active_card_ids': active, 'parent_sha256': source_hash(parent_source)})
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
                    if reply.input_tokens > PROMPT_CAP:
                        raise RuntimeError('provider reported input beyond the fixed prompt limit')
                    if reply.error:
                        raise ValueError(reply.error)
                    proposal = parse_proposal(reply.text)
                record['source'] = proposal.source
                prediction = _prediction_dict(proposal.prediction)
                record['prediction'] = prediction
                selected = tuple(prediction['selected_card_ids'])
                unfold_next = tuple(prediction.get('unfold_ids', ()))
                record['selected_card_ids'] = list(selected)
                if any(key not in active for key in selected):
                    raise ValueError('selection includes a card that was not displayed')
                if any(key not in {row['cand_id'] for row in self.ledger.records} for key in unfold_next):
                    raise ValueError('unknown unfold node')
                record['unfold_ids'] = list(unfold_next)
                if LABELS.search(proposal.source):
                    raise ValueError('proposal uses a reserved environment label')
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
                            record['outcome'] = promotion_decision(parent['aggregate'], candidate['aggregate'], paired)
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
                cooldown = advance_cooldown(cooldown, record['selected_card_ids'],
                                            record['outcome'] != 'PROMOTED' and _adverse(record['deltas']))
                record['cooldown_after'] = cooldown
                self.ledger.append(record)
            print(f'ROUND {self.call_prefix}/{cid} {record["outcome"]}', flush=True)
            if record['outcome'] == 'PROMOTED':
                parent_id, parent_source, parent = cid, record['source'], record['evaluation']
            unfold = tuple(record['unfold_ids'])
        return {'final_id': parent_id, 'final_sha256': source_hash(parent_source), 'rounds': self.rounds,
                'records': self.ledger.records}
