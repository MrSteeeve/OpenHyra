from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from typing import Any, Iterable, Mapping

import numpy as np

from schemas_v5 import BehaviorProfile


class BehaviorIndex:
    """Assign behavior cells and perform normalized nearest-neighbor search.

    The original V5 index is deliberately kept as a two-dimensional
    ``(performance, tail_risk)`` index.  ``assign_atlas_cell`` layers the
    mechanism and regime labels around that legacy cell, which gives callers
    a small quality-diversity archive without changing the persisted
    ``BehaviorProfile`` schema.  Missing labels are explicit ``"unknown"``
    values rather than silently collapsing records into a global best bucket.
    """

    def __init__(self, bucket_boundaries: dict[str, list[float]]) -> None:
        missing = {"performance", "tail_risk"}.difference(bucket_boundaries)
        if missing:
            raise ValueError(f"missing bucket boundaries: {', '.join(sorted(missing))}")
        self.bucket_boundaries = {
            name: sorted(float(value) for value in boundaries)
            for name, boundaries in bucket_boundaries.items()
        }

    def assign_cell(self, profile: BehaviorProfile) -> tuple[int, int]:
        performance = float(profile.performance["paired_mean"])
        tail_risk = float(profile.outcome_distribution["cvar_95"])
        return (
            bisect_right(self.bucket_boundaries["performance"], performance),
            bisect_right(self.bucket_boundaries["tail_risk"], tail_risk),
        )

    def assign_atlas_cell(
        self,
        profile: BehaviorProfile,
        *,
        mechanism_id: str | None = None,
        regime: str | None = None,
    ) -> tuple[str, tuple[int, int], str]:
        """Return a hashable ``mechanism × behavior × regime`` cell key.

        ``mechanism_id`` and ``regime`` are metadata owned by the harness (and
        therefore optional for archived legacy records).  Keeping the old
        behavior cell nested makes it unambiguous for consumers that still
        understand only the two numeric dimensions.
        """
        mechanism = str(mechanism_id).strip() if mechanism_id else "unknown"
        regime_label = str(regime).strip() if regime else "unknown"
        return mechanism, self.assign_cell(profile), regime_label

    # A descriptive alias is useful to integrations that call the structure
    # an ``atlas`` rather than an index.  It intentionally returns the same
    # tuple and does not create another persistence format.
    atlas_cell = assign_atlas_cell
    cell_key = assign_atlas_cell

    @staticmethod
    def _entry_field(entry: object, name: str, default: Any = None) -> Any:
        if isinstance(entry, Mapping):
            return entry.get(name, default)
        return getattr(entry, name, default)

    def quality_diversity_archive(
        self,
        entries: Iterable[Mapping[str, Any] | object],
        *,
        direction: str = "max",
        keep_failures: bool = True,
    ) -> dict[tuple[str, tuple[int, int], str], dict[str, Any]]:
        """Keep one deterministic elite per atlas cell.

        Entries may be mappings or small objects with ``profile``/``score``
        attributes.  A failed or unscored entry is retained only when its cell
        has no successful elite; this preserves counterexamples for Context
        without allowing them to displace a scored solution.  The comparator
        follows the task direction, so minimisation tasks do not accidentally
        cull their best (lowest) cells.
        """
        if direction not in {"min", "max"}:
            raise ValueError("direction must be 'min' or 'max'")

        archive: dict[tuple[str, tuple[int, int], str], dict[str, Any]] = {}

        def score_key(value: Any) -> float | None:
            if value is None:
                return None
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            return number if np.isfinite(number) else None

        for raw in entries:
            profile = self._entry_field(raw, "profile")
            if profile is None:
                # Accept the common serialized form while keeping malformed
                # rows out of the archive rather than raising during report
                # generation.
                profile_data = self._entry_field(raw, "behavior_profile")
                if isinstance(profile_data, BehaviorProfile):
                    profile = profile_data
                elif isinstance(profile_data, Mapping):
                    try:
                        profile = BehaviorProfile.from_dict(dict(profile_data))
                    except (TypeError, ValueError):
                        profile = None
            mechanism = self._entry_field(raw, "mechanism_id")
            if mechanism is None:
                mechanism = self._entry_field(raw, "mechanism")
            regime = self._entry_field(raw, "regime")
            if isinstance(profile, BehaviorProfile):
                key = self.assign_atlas_cell(
                    profile,
                    mechanism_id=mechanism,
                    regime=regime,
                )
            else:
                # Reports may already carry a compact numeric behavior cell
                # without the full profile object.  Accept that projection so
                # archive construction remains useful on exported bundles.
                raw_cell = self._entry_field(raw, "behavior_cell")
                if (
                    not isinstance(raw_cell, (list, tuple))
                    or len(raw_cell) != 2
                ):
                    continue
                try:
                    behavior_cell = (int(raw_cell[0]), int(raw_cell[1]))
                except (TypeError, ValueError):
                    continue
                mechanism_label = str(mechanism).strip() if mechanism else "unknown"
                regime_label = str(regime).strip() if regime else "unknown"
                key = (mechanism_label, behavior_cell, regime_label)
            candidate = dict(raw) if isinstance(raw, Mapping) else {
                name: getattr(raw, name)
                for name in ("record_id", "score", "status", "profile")
                if hasattr(raw, name)
            }
            if isinstance(profile, BehaviorProfile):
                candidate.setdefault("profile", profile)
            candidate_score = score_key(self._entry_field(raw, "score"))
            candidate_success = (
                candidate_score is not None
                and self._entry_field(raw, "status", "ok")
                in {"ok", "early_stopped", "success", None}
            )
            candidate["atlas_cell"] = key

            current = archive.get(key)
            if current is None:
                if candidate_success or keep_failures:
                    archive[key] = candidate
                continue

            current_score = score_key(current.get("score"))
            current_success = (
                current_score is not None
                and current.get("status", "ok")
                in {"ok", "early_stopped", "success", None}
            )
            # A valid score always outranks a failure.  Within the same class,
            # compare according to the task direction and break ties by the
            # stable record id so replay is deterministic.
            replace = False
            if candidate_success and not current_success:
                replace = True
            elif candidate_success == current_success:
                if candidate_score is not None and current_score is not None:
                    replace = (
                        candidate_score < current_score
                        if direction == "min"
                        else candidate_score > current_score
                    )
                    if candidate_score == current_score:
                        replace = str(candidate.get("record_id", "")) < str(
                            current.get("record_id", "")
                        )
                elif candidate_score is None and current_score is None:
                    replace = str(candidate.get("record_id", "")) < str(
                        current.get("record_id", "")
                    )
            if replace:
                archive[key] = candidate

        return archive

    # ``build_elite_archive`` reads naturally at call sites and is retained as
    # a compatibility alias for the quality-diversity operation.
    build_elite_archive = quality_diversity_archive
    elite_archive = quality_diversity_archive

    def behavior_vector(self, profile: BehaviorProfile) -> list[float]:
        performance = profile.performance
        outcome = profile.outcome_distribution
        geometry = profile.policy_geometry
        sensitivity = profile.sensitivity
        robustness = profile.robustness
        compute = profile.compute

        vector = list(performance["per_instance_improvement"])
        vector.extend(geometry["exercise_rate_by_instance"])
        vector.extend(
            [
                performance["paired_mean"],
                performance["paired_standard_error"],
                outcome["mean_loss"],
                outcome["var_95"],
                outcome["cvar_95"],
                geometry["reference_boundary_agreement"],
                robustness["seed_instability"],
                compute["training_seconds"],
                compute["inference_microseconds_per_state"],
                compute["parameter_count"],
                sensitivity["moneyness"],
                sensitivity["volatility"],
                sensitivity["correlation"],
                sensitivity["time_to_maturity"],
                robustness["input_scale_invariance_error"],
                robustness["state_perturbation_lipschitz_proxy"],
                robustness["seed_instability"],
            ]
        )
        return [float(value) for value in vector]

    def nearest_neighbors(
        self,
        query: BehaviorProfile,
        candidates: list[BehaviorProfile],
        k: int = 5,
    ) -> list[tuple[int, float]]:
        if k <= 0 or not candidates:
            return []

        candidate_vectors = np.asarray(
            [self.behavior_vector(profile) for profile in candidates],
            dtype=float,
        )
        query_vector = np.asarray(self.behavior_vector(query), dtype=float)
        if candidate_vectors.ndim != 2 or candidate_vectors.shape[1] != query_vector.size:
            raise ValueError("all behavior vectors must have the same length")

        means = np.mean(candidate_vectors, axis=0)
        standard_deviations = np.std(candidate_vectors, axis=0)
        scales = np.where(standard_deviations == 0.0, 1.0, standard_deviations)
        normalized_candidates = (candidate_vectors - means) / scales
        normalized_query = (query_vector - means) / scales
        distances = np.linalg.norm(
            normalized_candidates - normalized_query,
            axis=1,
        )
        indices = sorted(
            range(len(candidates)),
            key=lambda index: (float(distances[index]), index),
        )[: min(k, len(candidates))]
        return [(index, float(distances[index])) for index in indices]

    def cell_diversity(
        self,
        profiles: list[BehaviorProfile],
        *,
        labels: list[Mapping[str, Any] | object] | None = None,
    ) -> dict[str, Any]:
        """Summarize occupied cells, optionally using atlas labels.

        The no-label return shape is unchanged for existing callers.  When
        labels are supplied, the additional ``atlas_*`` fields expose the
        mechanism/regime coverage without perturbing legacy two-dimensional
        statistics.
        """
        if labels is not None and len(labels) != len(profiles):
            raise ValueError("labels must align one-to-one with profiles")
        if labels is None:
            counts = Counter(self.assign_cell(profile) for profile in profiles)
        else:
            counts = Counter(
                self.assign_atlas_cell(
                    profile,
                    mechanism_id=self._entry_field(label, "mechanism_id", self._entry_field(label, "mechanism")),
                    regime=self._entry_field(label, "regime"),
                )
                for profile, label in zip(profiles, labels)
            )
        cell_counts = list(counts.values())
        result = {
            "total_profiles": len(profiles),
            "occupied_cells": len(counts),
            "max_cell_count": max(cell_counts, default=0),
            "min_cell_count": min(cell_counts, default=0),
            "singleton_cells": sum(count == 1 for count in cell_counts),
        }
        if labels is not None:
            result.update({
                "atlas_occupied_cells": len(counts),
                "atlas_cells": [
                    {
                        "mechanism_id": key[0],
                        "behavior_cell": list(key[1]),
                        "regime": key[2],
                        "count": counts[key],
                    }
                    for key in sorted(counts, key=str)
                ],
            })
        return result
