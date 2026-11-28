"""Independent report checks on immutable prior results and offline card counts."""
from pathlib import Path

from experiments.calibrated_control_report import (
    _calibration_table, _env_tracks, _mean_sd, _old_tracks, _shape_error, posterior,
)

ROOT = Path(__file__).resolve().parents[1]


def test_historical_shape_and_gate_counts() -> None:
    tracks = _old_tracks(ROOT / 'artifacts' / 'p2-m3-isomorphic-control')
    alpha = [row for track in _env_tracks(tracks, 'env_alpha') for row in track]
    beta = [row for track in _env_tracks(tracks, 'env_beta') for row in track]
    assert len(alpha) == len(beta) == 18
    assert sum(_shape_error(row) for row in alpha) == 15
    assert sum(_shape_error(row) for row in beta) == 0
    assert sum('C06' in row['active_card_ids'] for row in alpha) == 0
    assert sum('C06' in row['active_card_ids'] for row in beta) == 14


def test_prior_posterior_counts_only_valid_cited_candidates() -> None:
    def row(interval: list[float] | None, status: str, cards: list[str]) -> dict:
        return {'evaluation': {'status': status}, 'selected_card_ids': cards,
                'prediction': {'target_metric': 'dt_nll'},
                'deltas': {'dt_nll': {'interval': interval}}}
    rows = [{'outcome': 'SEED'}, row([-2., -1.], 'VALID', ['C06', 'C08']),
            row([1., 2.], 'VALID', ['C06']), row([-1., 1.], 'VALID', ['C06']),
            row(None, 'VALID', ['C06']), row([3., 4.], 'GATE_REJECTED', ['C06'])]
    result = posterior({'rep_0/env_alpha/true_z_d': rows})
    assert result['env_alpha']['C06'] == {'n_valid': 4, 'n_plus': 1, 'n_minus': 1,
                                          'n_inconclusive': 2, 'posterior_mean': .5}
    assert result['env_alpha']['C08']['posterior_mean'] == 2/3
    assert result['env_beta']['C06']['posterior_mean'] == .5


def test_replicate_summary_reports_missing_coverage() -> None:
    assert _mean_sd([0., 1., None]).endswith('(2/3)')
    assert _mean_sd([None, None, None]) == 'NA (0/3)'
