#!/usr/bin/env python3
"""Run a frozen, two-round mechanism exploration pilot with real LLM calls.

Not a one-factor ablation: source exposure and exploration instructions change
together. Numerical evaluations use the unmodified trusted sandbox/evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import proposal_agent
from llm_backend import run_agent
from research_planning import canonical, digest, parse_plan, planning_prompt, research_packet, selected_plan
from sandbox import NUMERIC_THREAD_ENV, snapshot_source_tree, source_tree_hash

EVALUATOR = ROOT / "tasks/bermudan_optimal_stopping/evaluator.py"
SEED_SOURCE = ROOT / "tasks/bermudan_python_search/research_candidates/linear_ridge"
ARMS = ("parent_visible", "mechanism_first")


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    """Exclusive writes: an interrupted attempt cannot be silently rerolled."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def run_secret(out):
    """One private pricing salt and audit-seed offset per experiment output.

    Shared by every evaluation (common random numbers across arms); kept out of
    protocol.json and prompts. Created once, before any evaluation.
    """
    path = Path(out) / "private" / "evaluation_secret.json"
    if not path.exists():
        save(path, dict(pricing_salt=secrets.randbits(64), audit_seed_offset=secrets.randbits(62)))
    return load(path)


def request(seed, private=False, secret=None):
    """``secret=None`` gives the public protocol template (no salt, no private seed)."""
    config = dict(instance_count=2, repeats=1, training_paths=256,
                  pricing_paths=1024, outer_paths=128, inner_paths=8,
                  training_timeout_s=30, independent_validation=private, research_mode=True)
    if secret is not None:
        config["pricing_salt"] = secret["pricing_salt"]
    if private:
        seed = None if secret is None else secret["audit_seed_offset"] + seed
    else:
        seed = 770000 + seed
    return dict(schema="openhyra-evaluation-request.v1", stage="audit",
                protocol="bermudan-python-program-search.v1", task="bermudan_python_search",
                suite_id="bermudan-python-hidden-v2", seed=seed, config=config)


def frozen_files():
    paths = [p for p in ROOT.rglob("*.py")
             if not any(x in p.relative_to(ROOT).parts for x in ("artifacts", ".git", "__pycache__", ".venv"))]
    paths += [ROOT / "tasks/bermudan_python_search/TASK.md", SEED_SOURCE / "manifest.json"]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def verify(protocol):
    for rel, expected in protocol["source_hashes"].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f"Frozen code changed: {rel}")
    if source_tree_hash(SEED_SOURCE, 1024 * 1024)[0] != protocol["seed_source_digest"]:
        raise RuntimeError("Seed source changed")


def without_salt(req):
    """Saved request copies omit the salt; it lives only in private/evaluation_secret.json."""
    return {**req, "config": {k: v for k, v in req["config"].items() if k != "pricing_salt"}}


def evaluate(source, req, dest):
    h = source_tree_hash(source, 1024 * 1024)[0]
    save(dest.with_suffix(".request.json"), without_salt(req))
    start = time.monotonic()
    try:
        # The salted request reaches the evaluator only via a transient 0600 file.
        with tempfile.NamedTemporaryFile("w", suffix=".json") as salted:
            salted.write(json.dumps(req))
            salted.flush()
            r = subprocess.run([sys.executable, str(EVALUATOR), str(source), salted.name],
                               cwd=ROOT, env={**os.environ, **NUMERIC_THREAD_ENV},
                               capture_output=True, text=True, timeout=480)
        dest.with_suffix(".stderr.txt").write_text(r.stderr)
        payload = json.loads(r.stdout)
        if r.returncode or "error" in payload:
            raise RuntimeError(str(payload.get("error", r.stderr[-2000:])))
        value = dict(status="ok", score=payload["score"], metrics=payload["metrics"],
                     evidence=payload["evidence"], failure=None)
    except Exception as exc:
        value = dict(status="error", score=None, metrics={}, evidence={}, failure=f"{type(exc).__name__}: {exc}"[-4000:])
    if source_tree_hash(source, 1024 * 1024)[0] != h:
        raise RuntimeError("Candidate source mutated")
    value.update(source_digest=h, request=without_salt(req), evaluator_sha256=sha(EVALUATOR), wall_seconds=time.monotonic() - start)
    save(dest, value)
    print(f"EVAL {dest.name}: {value['status']} score={value['score']}", flush=True)
    return value


class Calls:
    def __init__(self, directory, model):
        self.directory = directory
        self.model = model
        self.count = 0

    def __call__(self, prompt, **kwargs):
        index = self.count
        self.count += 1
        role = "proposal" if kwargs.get("writable") else "context"
        save(self.directory / f"{index:02d}-input.json", dict(prompt=prompt, role=role, model=self.model,
             timeout_s=kwargs["timeout_s"], backend="codex"))
        start = time.monotonic()
        try:
            r = run_agent(prompt, **kwargs)
        except subprocess.TimeoutExpired as exc:
            def text(x):
                return x.decode(errors="replace") if isinstance(x, bytes) else x or ""
            r = subprocess.CompletedProcess([], 124, text(exc.stdout), text(exc.stderr))
        except Exception as exc:
            r = subprocess.CompletedProcess([], 125, "", f"{type(exc).__name__}: {exc}")
        save(self.directory / f"{index:02d}-output.json", dict(returncode=r.returncode,
             stdout=r.stdout, stderr=r.stderr, wall_seconds=time.monotonic() - start))
        print(f"CALL {role}: {r.returncode} {time.monotonic()-start:.1f}s", flush=True)
        return r


def public_record(identifier, result, mechanism, structural_claim=None):
    # Retain numerical feedback without copying opaque paths or private artifacts.
    return dict(id=identifier, data_role="public", status=result["status"], score=result["score"],
                failure=result["failure"], source_digest=result["source_digest"], mechanism=mechanism,
                structural_claim=structural_claim, observations=result["metrics"].get("summaries", []))


def run(out, seed, model):
    out.mkdir(parents=True, exist_ok=False)
    secret = run_secret(out)
    contract = (ROOT / "tasks/bermudan_python_search/TASK.md").read_text().split("## Search semantics")[0]
    protocol = dict(schema="openhyra-innovation-transition-pilot.v1", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        source_hashes=frozen_files(), git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        seed=seed, model=model, backend="codex", rounds=2, arms=ARMS,
        context_timeout_s=180, proposal_timeout_s=300, repairs_per_attempt=0,
        public_request=request(seed), private_request=request(seed, True),
        seed_source_digest=source_tree_hash(SEED_SOURCE,1024*1024)[0],
        objective="At least one generated, executable substantive change from the Ridge seed; superiority not required.",
        design="Bundled exploratory intervention: mechanism-first planning without parent source plus explicit nonlocal portfolio selection. Parent-visible comparison has same three-plan schema and call/evaluation caps. Not isolated source-ablation, spontaneous novelty, or population effect.",
        feedback="Each arm receives its own completed public candidates including failures and non-incumbents. Seed stays fixed to isolate planning from parent selection. No private feedback.",
        selection="Execute Context selected_id. No family whitelist, score filter, retry or manual plan replacement. Evaluate all successful candidates privately only after both arms finish.",
        success="Independent source/dataflow inspection plus successful fit/predict; self-declared families never establish transition.",
        cost="Two Context and at most two Proposal calls per arm; actual token/time differences reported. One independent seed; no stability claim.",
        python=sys.version, numpy=np.__version__)
    save(out / "protocol.json", protocol)
    shutil.copytree(SEED_SOURCE, out / "seed")
    # Explicit writable editing copy; the immutable seed snapshot is never edited.
    for p in (out / "seed").rglob("*"):
        if p.is_file():
            p.chmod(p.stat().st_mode | 0o200)
    baseline = evaluate(out / "seed", request(seed, secret=secret), out / "baseline-public.json")
    if baseline["status"] != "ok":
        raise RuntimeError("Baseline evaluator preflight failed; no model calls issued")
    initial = public_record("ridge-seed", baseline, "Ridge backward continuation regression")
    histories = {a: [initial] for a in ARMS}
    rows = []
    for round_index in range(2):
        for arm in (ARMS if round_index == 0 else tuple(reversed(ARMS))):
            verify(protocol)
            if source_tree_hash(out / "seed", 1024 * 1024)[0] != protocol["seed_source_digest"]:
                raise RuntimeError("Editing seed changed")
            identifier = f"{arm}-r{round_index}"
            directory = out / identifier
            directory.mkdir()
            packet = research_packet(histories[arm])
            save(directory / "packet.json", packet)
            prompt = planning_prompt(arm=arm, contract=contract, packet=packet,
                                     parent_source=(out / "seed/algorithm.py").read_text(), round_index=round_index)
            calls = Calls(directory / "calls", model)
            workspace = directory / "context_workspace"
            workspace.mkdir()
            plan = None
            chosen = None
            metadata = {}
            try:
                result = calls(prompt, cwd=workspace, writable=False, timeout_s=180, backend="codex", model=model)
                if result.returncode:
                    raise RuntimeError(f"Context exited {result.returncode}")
                plan = parse_plan(result.stdout, packet)
                save(directory / "plan.json", plan)
                chosen = selected_plan(plan)
                proposal_prompt = contract + "\nImplement this selected research plan faithfully:\n" + canonical(chosen) + """
Use only NumPy and standard library. Fit cap 30 seconds, 256 training paths.
Do not execute candidate code here, inspect other directories, or use network.
Implement fit/predict CLI exactly. Training payoffs are ALREADY discounted;
do not discount again. Create output directories. Prediction must use current
time_index and causal inputs. Save numerical training diagnostics in the model
directory (iterations, fitted structure, objective/change where applicable) so
the actual learning mechanism can later be inspected. Do not silently substitute
the seed algorithm. Keep implementation compact and runnable.
"""
                original = proposal_agent.run_agent
                proposal_agent.run_agent = calls
                try:
                    ok, description = proposal_agent.propose(out / "seed", directory / "draft", proposal_prompt,
                        ["algorithm.py", "manifest.json"], timeout_s=300, backend="codex", model=model,
                        candidate_mode="python_program", entrypoint="algorithm.py", source_files=["algorithm.py","manifest.json"],
                        intervention=dict(intervention_operator="whole_program_restart",intervention_scope="whole_program",
                                          mechanism=chosen["mechanism"],prediction=chosen["prediction"],falsifier=chosen["falsifier"]),
                        execution_metadata=metadata)
                finally:
                    proposal_agent.run_agent = original
                if not ok:
                    raise RuntimeError(description)
                snapshot_source_tree(directory / "draft", directory / "program", 1024*1024)
                ev = evaluate(directory / "program", request(seed, secret=secret), directory / "public.json")
            except Exception as exc:
                ev = dict(status="error", score=None, metrics={}, evidence={}, source_digest=None,
                          failure=f"{type(exc).__name__}: {exc}"[-4000:])
            record = public_record(identifier, ev, chosen["mechanism"] if chosen else "no valid plan",
                                   chosen["structural_difference"] if chosen else None)
            save(directory / "record.json", dict(**record, operator_execution=metadata,
                 plan_digest=digest(plan), packet_digest=digest(packet), parent_id="ridge-seed"))
            histories[arm].append(record)
            rows.append(record)
    save(out / "public_complete.json", dict(records=rows, frozen_before_private=True))
    for identifier in ["seed"] + [r["id"] for r in rows if r["status"] == "ok"]:
        verify(protocol)
        source = out / "seed" if identifier == "seed" else out / identifier / "program"
        evaluate(source, request(seed, True, secret), out / f"private-{identifier}.json")
    save(out / "complete.json", dict(records=rows, success_requires_separate_mechanism_audit=True))
    save(out / "artifact_hashes.json", {str(p.relative_to(out)):sha(p) for p in out.rglob("*")
                                        if p.is_file() and not p.is_symlink()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--model", default="gpt-5.6-sol")
    args = parser.parse_args()
    run(args.output.resolve(), args.seed, args.model)
