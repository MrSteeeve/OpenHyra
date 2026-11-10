"""Reviewable source snapshots and genuinely blind human-review exports."""
from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import random
import subprocess
import sys
import tarfile

import scipy


_SOURCE_SUFFIXES = {".py", ".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".json",
                    ".js", ".ts", ".tsx", ".jsx", ".sh", ".tex", ".bib", ".dsl"}
_EXCLUDED_PARTS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__",
                   "artifacts", "artifact", "data", "datasets",
                   "secrets", "credentials", "keys", ".ssh", ".aws"}
_SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".crt", ".cer", ".kdbx"}
_DATA_SUFFIXES = {".csv", ".parquet", ".npy", ".npz", ".pkl", ".pickle", ".db", ".sqlite", ".h5"}


def _safe_path(path: str) -> bool:
    p = Path(path)
    lower = p.name.lower()
    if p.is_absolute() or ".." in p.parts:
        return False
    if any(part.lower() in _EXCLUDED_PARTS or part.lower().startswith((".venv", ".env")) for part in p.parts):
        return False
    if p.suffix.lower() in _SECRET_SUFFIXES | _DATA_SUFFIXES or lower.startswith(".env"):
        return False
    if p.stem.lower() in {"key", "keys", "id_rsa", "id_ed25519"}:
        return False
    if any(token in lower for token in ("secret", "credential", "token", "password", "api_key", "apikey", "private_key", "id_rsa", "id_ed25519")):
        return False
    return True


def _safe_source(path: str) -> bool:
    p = Path(path)
    return _safe_path(path) and (p.suffix.lower() in _SOURCE_SUFFIXES or p.name.lower() in {"license", "makefile", "dockerfile"})


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def _dependency_versions() -> dict:
    versions = {}
    for dependency in ("numpy", "scipy", "torch", "structurerdsl", "httpx"):
        try:
            versions[dependency] = metadata.version(dependency)
        except metadata.PackageNotFoundError:
            versions[dependency] = None
    return versions


def capture_provenance(output, repo_roots) -> dict:
    """Capture three repository revisions plus safe source and tracked diffs.

    repo_roots accepts {label: root} or a sequence of three roots. Excluded files
    are only classified by path, never opened. Symlinks are not dereferenced.
    Source archive includes tracked and relevant untracked sources, including
    Stochopia adapters. Tracked diff covers all nonexcluded source paths.
    """
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    roots = repo_roots if isinstance(repo_roots, dict) else {Path(r).name: r for r in repo_roots}
    if len(roots) != 3:
        raise ValueError("Exactly three repository roots are required")
    if any(not str(label).replace("-", "").replace("_", "").isalnum() for label in roots):
        raise ValueError("Repository labels must be simple names")
    manifest = {"schema": "path-research-provenance-v1", "scipy_version": scipy.__version__,
                "dependency_versions": _dependency_versions(), "python_executable": sys.executable,
                "source_archive": str(output / "source_snapshot.tar.gz"), "repositories": {},
                "exclusion_policy": "No env, keys, credentials, venv, artifacts, data or symlink contents are read or copied; public source settings are included"}
    with tarfile.open(output / "source_snapshot.tar.gz", "w:gz") as archive:
        for label, root in roots.items():
            root = Path(root).resolve()
            tracked = _git(root, "ls-files", "-z").decode().split("\0")
            untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
            all_paths = sorted(set(filter(None, tracked + untracked)))
            included, excluded = [], []
            for source in all_paths:
                path = root / source
                if (_safe_source(source) and path.is_file() and not path.is_symlink()
                        and not output.is_relative_to(path) and not path.is_relative_to(output)):
                    included.append(source)
                    archive.add(path, arcname=f"{label}/{source}", recursive=False)
                else:
                    excluded.append(source)
            tracked_safe = sorted(source for source in tracked if source and _safe_path(source))
            # One tracked diff against HEAD includes both staged and unstaged work.
            diff = _git(root, "diff", "HEAD", "--binary", "--", *tracked_safe) if tracked_safe else b""
            diff_path = output / f"{label}.tracked.diff"
            diff_path.write_bytes(diff)
            manifest["repositories"][str(label)] = {
                "root": str(root), "commit_sha": _git(root, "rev-parse", "HEAD").decode().strip(),
                "dirty": bool(_git(root, "status", "--porcelain", "--untracked-files=all")),
                "tracked_diff": str(diff_path), "tracked_diff_paths": tracked_safe,
                "source_paths": included,
                "untracked_source_paths": [p for p in included if p in untracked],
                "excluded_paths": excluded}
    manifest_path = output / "manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


_SCIENTIFIC_SOURCES = {
    "stochopia": ("stochopia/research/",),
    "structurerdsl": ("structurerdsl/",),
    "openhyra": (
        "tasks/stochopia_structuring/theoretical_frontier.py",
        "tasks/stochopia_structuring/path_research_protocol.py",
        "tasks/stochopia_structuring/known_path_families.py",
        "tasks/stochopia_structuring/generator.py",
        "tasks/stochopia_structuring/api_config.py",
        "tasks/stochopia_structuring/api_models.json",
        "tasks/stochopia_structuring/economic_behavior.py",
        "tasks/stochopia_structuring/controlled_statistics.py",
        "tasks/stochopia_structuring/research_provenance.py",
        "experiments/run_stochopia_path_research.py",
    ),
}


def verify_provenance(manifest_path, source_prefixes=None) -> dict:
    """Require the same three commits and selected scientific source bytes.

    Call immediately before experiment generation. Uncommitted source additions,
    edits and deletions are compared against the preflight's saved source archive,
    with no hashes. Optional source_prefixes maps repository labels to relative
    directories or exact file paths. Default selection covers Stochopia research,
    StructurerDSL source and OpenHyra's path-study evaluators/generator/entry point.
    Secret paths are excluded before any contents are read. Unrelated dirty files
    do not block this experiment-version consistency check.
    """
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repositories = manifest["repositories"]
    if len(repositories) != 3:
        raise ValueError("Preflight provenance must contain exactly three repositories")
    mismatches, checked, commits = [], {}, {}
    archive_path = Path(manifest["source_archive"])
    if not archive_path.is_absolute():
        archive_path = manifest_path.parent / archive_path
    with tarfile.open(archive_path, "r:gz") as archive:
        for label, record in repositories.items():
            root = Path(record["root"]).resolve()
            current_commit = _git(root, "rev-parse", "HEAD").decode().strip()
            commits[label] = current_commit
            if current_commit != record["commit_sha"]:
                mismatches.append(f"{label}: commit differs from preflight")
            prefixes = (source_prefixes.get(label, ()) if source_prefixes is not None
                        else _SCIENTIFIC_SOURCES.get(label.lower(), ()))
            if isinstance(prefixes, str):
                prefixes = (prefixes,)
            prefixes = tuple(str(prefix).rstrip("/") for prefix in prefixes)

            def selected(path):
                return (_safe_source(path) and Path(path).suffix in {".py", ".json"}
                        and any(path == prefix or path.startswith(prefix + "/") for prefix in prefixes))

            stored = {path for path in record["source_paths"] if selected(path)}
            current_paths = _git(root, "ls-files", "-z").decode().split("\0")
            current_paths += _git(root, "ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
            current = {path for path in current_paths if path and selected(path)
                       and (root / path).is_file() and not (root / path).is_symlink()}
            checked[label] = sorted(stored | current)
            for path in sorted(stored - current):
                mismatches.append(f"{label}/{path}: scientific source removed or replaced")
            for path in sorted(current - stored):
                mismatches.append(f"{label}/{path}: scientific source added since preflight")
            for path in sorted(stored & current):
                member = archive.getmember(f"{label}/{path}")
                saved = archive.extractfile(member).read()
                if saved != (root / path).read_bytes():
                    mismatches.append(f"{label}/{path}: scientific source modified since preflight")
    if mismatches:
        raise ValueError("Experiment source differs from preflight: " + "; ".join(mismatches))
    return {"verified": True, "manifest_path": str(manifest_path),
            "commit_shas": commits, "checked_source_paths": checked,
            "method": "Literal source-byte comparison to preflight archive; three commit comparisons"}


_ASSESSMENT_FIELDS = {
    "feasible_under_stated_constraints": None,
    "economic_mechanism_description": None,
    "matches_known_family": None,
    "known_family_name_and_reference": None,
    "distinct_mechanism_supported": None,
    "hedging_or_execution_concerns": None,
    "reviewer_confidence": None,
    "reviewer_notes": None,
}


def export_blind_review(candidates, output, *, seed=20261012) -> dict:
    """Export shuffled structured cards plus a separate private origin mapping.

    Candidate payload is allowlisted: demand, market, and cashflows. These must
    be numerical/public structured values, not free-form model commentary.
    Reviewers must complete the null assessment fields; no AI review is supplied.
    Full provenance remains only in private_mapping.json, which must not be
    given to reviewers. This is source blinding, not a promise that recognizable
    economic structure cannot suggest an origin to a human.
    """
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates = list(candidates)
    order = list(range(len(candidates)))
    random.Random(seed).shuffle(order)
    cards, mapping = [], {}
    for number, index in enumerate(order, 1):
        candidate = candidates[index]
        card_id = f"card-{number:04d}"
        payload = _strip_origin({key: candidate[key] for key in ("demand", "market", "cashflows") if key in candidate})
        if not payload:
            raise ValueError("Blind review requires structured economic payload")
        _validate_blind_payload(payload)
        cards.append({"card_id": card_id, "economic_payload": payload,
                      "assessment": dict(_ASSESSMENT_FIELDS), "review_status": "awaiting_human"})
        # Preserve removed nested origins too, exclusively in the private file.
        mapping[card_id] = {"source_index": index, "original_candidate": candidate}
    cards_path, private_path = output / "review_cards.json", output / "private_mapping.json"
    cards_path.write_text(json.dumps({"schema": "blind-human-review-v1", "cards": cards,
                                    "expert_review_complete": False}, indent=2), encoding="utf-8")
    private_path.write_text(json.dumps({"seed": seed, "card_origin_mapping": mapping}, indent=2), encoding="utf-8")
    private_path.chmod(0o600)
    return {"review_cards": str(cards_path), "private_mapping": str(private_path),
            "n_cards": len(cards), "expert_review_complete": False,
            "human_review_required": True}


_ORIGIN_KEYS = {"model", "model_group", "arm", "parent", "parent_id", "round", "rationale",
                "candidate_id", "pair_id", "origin", "provenance", "generator", "seed"}


def _origin_key(key):
    return str(key).lower() in _ORIGIN_KEYS or any(t in str(key).lower() for t in ("rationale", "provenance", "model", "parent"))


def _strip_origin(value):
    if isinstance(value, dict):
        return {key: _strip_origin(item) for key, item in value.items() if not _origin_key(key)}
    if isinstance(value, (list, tuple)):
        return [_strip_origin(item) for item in value]
    return value


_BLIND_ENUMS = {
    "schema": {"stochopia-path-demand.v1", "stochopia-path-demand.v2", "stochopia-shared-vol-tree.v1"},
    "direction": {"below", "above"},
    "monitoring": {"monthly"},
    "funding_mode": {"minimum_service_capital", "fixed_budget_leg_control"},
    "scope": {"finite binomial model; shared diffusion sigma, not identical finite-step P/Q variances"},
}


def _validate_blind_payload(value, field=None):
    """Validate purely economic content after removing nested provenance keys."""
    if isinstance(value, dict):
        for key, item in value.items():
            if _origin_key(key):
                raise ValueError("Origin information in blind payload")
            _validate_blind_payload(item, field=key)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_blind_payload(item, field=field)
    elif isinstance(value, str):
        # Free-text narrative can reveal model/arm/rationale. Public numerical
        # structure can use schema keys, but no string-valued commentary.
        if value not in _BLIND_ENUMS.get(field, set()):
            raise ValueError("Blind economic payload must be numerical structured data or fixed factual schema/direction enums")
    elif value is not None and not isinstance(value, (int, float, bool)):
        raise ValueError("Unsupported blind payload value")
