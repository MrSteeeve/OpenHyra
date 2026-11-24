"""Publish compact, reproducible tables alongside the sealed M4 JSON."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable
from pathlib import Path

from stochopia.common import atomic_write_text

from .formal_verify import load_frozen_study
from .store import IMPACT, NLL, Json, Ledger


TRIAL_FIELDS = (
    "stock", "arm", "search_seed", "round", "trial_id", "verdict", "parent_id", "candidate_id",
    "source_sha256", "compiled", "compiler_repairs", "http_attempts", "input_tokens", "output_tokens",
    "api_seconds", "training_budget_consumed", "val_nll_total", "lob_impact_score", "mean_delta_score",
    "p05_delta_score", "primary", "guardrail",
)
CALL_FIELDS = (
    "stock", "arm", "search_seed", "round", "trial_id", "request", "stage", "http_attempt", "completed",
    "input_tokens", "output_tokens", "seconds", "retryable", "diagnostic",
)
COMPILE_FIELDS = (
    "stock", "arm", "search_seed", "round", "trial_id", "stage", "attempt",
    "source_sha256", "accepted", "diagnostic",
)
RESULT_FIELDS = (
    "stock", "arm", "search_seed", "phase", "task_id", "model_id", "status", "val_nll_total",
    "lob_impact_score", "impact_ci95_low", "impact_ci95_high", "impact_delta_from_initial",
    "nll_delta_from_initial", "decision", "primary", "guardrail", "C1", "C2", "C3", "C4", "C5",
)
CROSS_FIELDS = (
    "arm", "source_stock", "target_stock", "task_id", "model_id", "source_sha256", "role",
    "status", "val_nll_total", "lob_impact_score", "impact_ci95_low", "impact_ci95_high",
)
AUDIT_FIELDS = (
    "stock", "arm", "task_id", "status", "test_decision", "mean_delta_score", "p05_delta_score",
    "has_benchmark_gain", "primary", "guardrail",
)


def _write_csv(path: Path, columns: tuple[str, ...], rows: Iterable[Json]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _roles(assessment: Json) -> Json:
    return {item["role"]: item["status"] for item in assessment.get("prediction_verdicts", [])}


def _diagnostic(value: str | None) -> str | None:
    if value is None:
        return None
    lines = value.strip().splitlines()
    if not lines:
        return None
    excerpt = lines[-1][:400]
    return "'" + excerpt if excerpt.startswith(("=", "+", "-", "@")) else excerpt


def _request_rows(trial: Json, metadata: Json) -> list[Json]:
    requests = {item["record_id"]: item for item in trial["requests"]}
    rows = []
    for name, reservation in requests.items():
        if name.endswith("-result"):
            continue
        result = requests.get(f"{name}-result")
        fields = {}
        if result is not None:
            fields = {
                key: result.get(key) for key in ("input_tokens", "output_tokens", "seconds", "retryable")
            }
            fields["diagnostic"] = _diagnostic(result.get("error"))
        rows.append({
            **metadata, **reservation, **fields, "request": name, "completed": result is not None,
        })
    return rows


def _development_rows(manifest: Json) -> tuple[list[Json], list[Json], list[Json]]:
    trials = []
    calls = []
    compilation = []
    for run in manifest["trajectories"].values():
        for trial in run["trials"]:
            metadata = {key: run[key] for key in ("stock", "arm", "search_seed")}
            metadata.update(round=trial["round"], trial_id=trial["trial_id"])
            attempts = _request_rows(trial, metadata)
            calls.extend(attempts)
            for attempt in trial["compiler"]:
                compilation.append({
                    **metadata, **attempt, "diagnostic": _diagnostic(attempt["diagnostic"]),
                    "accepted": attempt["diagnostic"] is None,
                })
            candidate = trial["candidate"] or {}
            trials.append({
                **metadata, **trial.get("candidate_metrics", {}), **_roles(trial),
                "verdict": trial["verdict"], "parent_id": trial["parent"]["model_id"],
                "candidate_id": candidate.get("model_id"), "source_sha256": candidate.get("source_sha256"),
                "compiled": trial["program"]["status"] == "ready",
                "compiler_repairs": sum(row["stage"] == "repair" for row in trial["compiler"]),
                "http_attempts": len(attempts),
                "input_tokens": sum(row.get("input_tokens") or 0 for row in attempts),
                "output_tokens": sum(row.get("output_tokens") or 0 for row in attempts),
                "api_seconds": sum(row.get("seconds") or 0 for row in attempts),
                "training_budget_consumed": trial["training_budget_consumed"],
                "mean_delta_score": trial.get("mean_delta_score"), "p05_delta_score": trial.get("p05_delta_score"),
            })
    return trials, calls, compilation


def _endpoint_rows(report: Json) -> Iterable[Json]:
    for trajectory in report["trajectories"].values():
        metadata = {key: trajectory[key] for key in ("stock", "arm", "search_seed")}
        for label in ("initial", "first", "final"):
            endpoint = trajectory[label]
            if endpoint is None:
                yield {**metadata, "phase": label, "status": "unavailable"}
                continue
            assessment = endpoint["direct_parent_assessment"] or {}
            metrics = endpoint["metrics"]
            ci = endpoint["impact_ci95"] or (None, None)
            yield {
                **metadata, **endpoint, **_roles(assessment), "phase": label,
                NLL: metrics.get(NLL), IMPACT: metrics.get(IMPACT), "decision": assessment.get("verdict"),
                "impact_ci95_low": ci[0], "impact_ci95_high": ci[1],
                **metrics.get("claims", {}),
            }


def _cross_row(arm: str, source: str, target: str, result: Json, role: str) -> Json:
    metrics = result["metrics"]
    interval = result.get("impact_ci95") or (None, None)
    return {
        **result, **result["source"], "arm": arm, "source_stock": source, "target_stock": target,
        "role": role, NLL: metrics.get(NLL), IMPACT: metrics.get(IMPACT),
        "impact_ci95_low": interval[0], "impact_ci95_high": interval[1],
    }


def _cross_rows(report: Json) -> Iterable[Json]:
    for arm, sources in report["cross_regime"].items():
        for source, data in sources.items():
            for target, result in data["cells"].items():
                yield _cross_row(arm, source, target, result, "selected")
            if data["auxiliary"] is not None:
                for field in ("diagonal", "off_diagonal"):
                    source_result = data["auxiliary"][field]
                    result = {
                        **source_result,
                        "source": source_result["candidate"],
                        "impact_ci95": source_result["readouts"].get(IMPACT, {}).get("candidate_interval"),
                    }
                    yield _cross_row(arm, source, source_result["stock"], result, "auxiliary")


def _audit_rows(report: Json) -> Iterable[Json]:
    for row in report["audit_only_rejected_unverified"]:
        assessment = row["assessment"]
        yield {
            **row, **assessment, **_roles(assessment),
            "arm": row["group"], "test_decision": assessment["verdict"],
        }


def _readme(manifest: Json, *, has_holdout: bool) -> str:
    code = manifest["code"]
    status = "留出复验已落盘。" if has_holdout else "开发记录与留出任务清单已封存。"
    return f"""# Stochopia × OpenHyra P1-M4 实验证据

{status} 每条轨迹的首次、末次和审计任务见 `study_manifest.json`。候选源码的 SHA-256 对应 `sources/<sha256>.py`。只有沙箱通过的候选才可进入复验。同一候选出现在多个登记角色时共用一次配对结果。

- `development-trials.csv` 汇总 90 次候选的编译、预算、护栏和双预测结果。
- `api-attempts.csv` 按请求记录 token、时长及 HTTP 重试。服务端未报告用量的单元格为空。
- `compiler-attempts.csv` 记录匹配源码与修复错误摘要。
- `verification/report.json` 与 `verification/results/` 保存冻结任务复验和断点记录。报告的 `audit_only_rejected_unverified` 是单独的证伪审计集。
- `verification/test-data-reconciliation.json` 记录测试日重构与窗口盘口核对。
- `holdout-endpoints.csv`、`rejected-audit.csv` 和 `cross-regime.csv` 汇总正式复验结果及跨股矩阵。备用架构在表中标为 `auxiliary`。

## 复现入口

Stochopia 锁码提交为 `{code.get("stochopia_commit", "")}`。OpenHyra 锁码提交为 `{code.get("openhyra_commit", "")}`。完整树标识、数据版本、日期和配对种子随清单登记。原始行情、含 API 原始响应的工作账本、训练权重与逐笔轨迹保存在仓库外。

从锁码版本、两仓的 Python 环境和获准的库外行情目录运行以下命令。开始下载留出交易日前，先完成开发搜索并确认封存清单。复验中断时沿原目录恢复。

```bash
export PYTHONPATH="$OPENHYRA:$STOCHOPIA"
"$STOCHOPIA/.venv/bin/python" -m tasks.stochopia_p1.formal dev \\
  --artifact-root "$ARTIFACTS" --market-root "$MARKET" --tokenizers-dir "$TOKENIZERS"
"$STOCHOPIA/.venv/bin/python" -m tasks.stochopia_p1.formal freeze \\
  --artifact-root "$ARTIFACTS" --market-root "$MARKET"
"$STOCHOPIA/.venv/bin/python" -m tasks.stochopia_p1.holdout_data \\
  --artifact-root "$ARTIFACTS" --market-root "$MARKET"
"$STOCHOPIA/.venv/bin/python" -m tasks.stochopia_p1.formal verify \\
  --artifact-root "$ARTIFACTS" --market-root "$MARKET"
```
"""


def write_archive_tables(root: Path) -> None:
    """Publish allowlisted JSON-derived CSV tables without reading market data."""
    manifest = load_frozen_study(root)
    ledger = Ledger(root)
    with ledger.lock():
        trials, calls, compilation = _development_rows(manifest)
        _write_csv(ledger.file("development-trials.csv"), TRIAL_FIELDS, trials)
        _write_csv(ledger.file("api-attempts.csv"), CALL_FIELDS, calls)
        _write_csv(ledger.file("compiler-attempts.csv"), COMPILE_FIELDS, compilation)
        report = ledger.read("verification", "report.json")
        if report is not None:
            _write_csv(ledger.file("holdout-endpoints.csv"), RESULT_FIELDS, _endpoint_rows(report))
            _write_csv(ledger.file("rejected-audit.csv"), AUDIT_FIELDS, _audit_rows(report))
            _write_csv(ledger.file("cross-regime.csv"), CROSS_FIELDS, _cross_rows(report))
        atomic_write_text(ledger.file("README.md"), _readme(manifest, has_holdout=report is not None))
