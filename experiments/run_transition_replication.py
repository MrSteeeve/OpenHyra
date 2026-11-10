#!/usr/bin/env python3
"""Paired model/seed factorial replication; run only from a frozen runtime.

No trusted evaluator changes. Each candidate attempt is a separate process.
The supervisor uses complete-round barriers and pauses on resource failures.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_innovation_transition as pilot
import proposal_agent
from llm_backend import _run_cli
from research_planning import canonical, digest, parse_plan, planning_prompt, selected_plan
from sandbox import snapshot_source_tree, source_tree_hash

MODELS = ('gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-terra', 'gpt-5.6-sol')
SEEDS = tuple(range(101, 111))
ARMS = ('visible_generic', 'hidden_generic', 'visible_explore', 'hidden_explore')
ROUNDS = 20

class ResourceUnavailable(RuntimeError):
    pass

def save(path, value):
    """Atomic exclusive publication; never replace an existing research record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with tmp.open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()

def packet(records):
    public = [r for r in records if r.get('data_role') == 'public']
    rows = []
    for r in public:
        row = {k:r.get(k) for k in ('id','status','score','mechanism','failure','structural_claim','source_digest')}
        # Same bounded projection in all arms; retain every previous attempt.
        row['observations'] = [{k:s.get(k) for k in ('dimension','payoff_type','lower_bound',
            'lower_bound_standard_error','confidence_gap','candidate_training_seconds')}
            for s in r.get('observations', [])]
        rows.append(row)
    return dict(schema='openhyra-replication-packet.v1', records=rows,
                record_ids=[r['id'] for r in rows], source_records_digest=digest(public))

def prompt_for(arm, contract, evidence, parent, index):
    if arm not in ARMS:
        raise ValueError('unknown arm')
    # Reuse exactly the original schema/contract, separate the two interventions.
    base = planning_prompt(arm='parent_visible', contract=contract, packet=evidence,
        parent_source='', round_index=index).split('\nPropose the most promising improvements to this parent.')[0]
    generic = '\nChoose the most promising feasible improvement. Local refinements and changes of method are both permitted.\n'
    explore = planning_prompt(arm='mechanism_first', contract=contract, packet=evidence,
        parent_source='', round_index=index).split('\nMechanism-first exploration:')[1]
    instruction = '\nMechanism-first exploration:' + explore if arm.endswith('_explore') else generic
    source = '\nParent implementation:\n' + parent if arm.startswith('visible_') else '\nParent implementation is not shown.\n'
    return base + instruction + source

def verify(out):
    protocol = pilot.load(out/'protocol.json')
    for rel, expected in protocol['source_hashes'].items():
        if pilot.sha(ROOT/rel) != expected:
            raise RuntimeError('Frozen runtime drift: '+rel)
    if source_tree_hash(out/'seed', 1024*1024)[0] != protocol['seed_source_digest']:
        raise RuntimeError('Seed source drift')
    if protocol['models'] != list(MODELS) or protocol['seeds'] != list(SEEDS) or protocol['arms'] != list(ARMS) or protocol['rounds'] != ROUNDS:
        raise RuntimeError('Design drift')
    return protocol

class Calls:
    def __init__(self, directory, model):
        self.directory, self.model = directory, model
    def __call__(self, prompt, **kwargs):
        self.directory.mkdir(parents=True, exist_ok=True)
        index = len(list(self.directory.glob('*-input.json')))
        if index and not (self.directory/f'{index-1:02d}-output.json').exists():
            raise ResourceUnavailable('Unfinished previous call requires explicit recovery; no silent reroll')
        role = 'proposal' if kwargs.get('writable') else 'context'
        cmd = ['codex','--ask-for-approval','never','exec','--ignore-user-config','--ephemeral',
               '--skip-git-repo-check','--color','never','--sandbox',
               'workspace-write' if kwargs.get('writable') else 'read-only',
               '-c','model_reasoning_effort="medium"','--model',self.model,'-']
        save(self.directory/f'{index:02d}-input.json', dict(prompt=prompt,role=role,model=self.model,
             reasoning_effort='medium',command=cmd,timeout_s=kwargs['timeout_s']))
        start = time.monotonic()
        try:
            result = _run_cli(cmd, cwd=kwargs['cwd'], prompt_stdin=prompt,
                              timeout_s=kwargs['timeout_s'], cancel_event=None)
        except subprocess.TimeoutExpired as exc:
            def decode(x): return x.decode(errors='replace') if isinstance(x,bytes) else x or ''
            result = subprocess.CompletedProcess(cmd,124,decode(exc.stdout),decode(exc.stderr))
        except OSError as exc:
            result = subprocess.CompletedProcess(cmd,125,'',str(exc))
        save(self.directory/f'{index:02d}-output.json', dict(returncode=result.returncode,
             stdout=result.stdout,stderr=result.stderr,wall_seconds=time.monotonic()-start))
        # Preserve account/network errors separately from scientific failures.
        if result.returncode not in (0,124):
            tail = (result.stderr+'\n'+result.stdout).lower()
            resource_words = ('usage limit','rate limit','quota','insufficient','not supported',
                              'unauthorized','authentication','connection','stream disconnected',
                              'failed to connect','access denied','not found')
            if result.returncode == 125 or any(w in tail for w in resource_words):
                raise ResourceUnavailable(f'{role} backend unavailable, retained call {index}')
        print(f'{role} {self.model} rc={result.returncode} seconds={time.monotonic()-start:.1f}',flush=True)
        return result

def refuse_unsalted_resume(out):
    """Runs begun before the per-run private pricing salt cannot be resumed; no migration."""
    if (out/'baselines').exists() and not (out/'private'/'evaluation_secret.json').exists():
        raise SystemExit(f'{out} was started before the per-run private pricing salt; resuming would mix '
                         'unsalted and salted evaluations. Start a fresh --output directory.')

def evaluate_once(source, request, dest):
    if dest.exists():
        result = pilot.load(dest)
        if result['source_digest'] != source_tree_hash(source,1024*1024)[0] or result['request'] != pilot.without_salt(request):
            raise RuntimeError('Cached evaluation provenance mismatch')
        return result
    if dest.with_suffix('.request.json').exists():
        raise ResourceUnavailable('Interrupted evaluator requires recovery; no silent reevaluation')
    return pilot.evaluate(source, request, dest)

def attempt(out, model, seed, arm, index):
    refuse_unsalted_resume(out)
    protocol = verify(out)
    root = out/'runs'/model/f's{seed}'/arm
    directory = root/f'r{index:02d}'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (directory/'record.json').exists(): return
        if (directory/'started.json').exists():
            raise ResourceUnavailable('Incomplete attempt retained; explicit recovery required before rerun')
        save(directory/'started.json',dict(pid=os.getpid(),started_unix=time.time()))
        initial = pilot.load(out/'baselines'/f's{seed}-record.json')
        history = [initial]+[pilot.load(root/f'r{i:02d}'/'record.json') for i in range(index)]
        evidence = packet(history)
        if not (directory/'packet.json').exists(): save(directory/'packet.json',evidence)
        elif pilot.load(directory/'packet.json') != evidence: raise RuntimeError('History drift')
        contract = (ROOT/'tasks/bermudan_python_search/TASK.md').read_text().split('## Search semantics')[0]
        calls = Calls(directory/'calls',model)
        workspace = directory/'context_workspace'
        workspace.mkdir(exist_ok=True)
        chosen = None
        plan = None
        metadata = {}
        stage = 'context'
        try:
            if (directory/'plan.json').exists():
                plan = parse_plan(canonical(pilot.load(directory/'plan.json')),evidence)
            else:
                response = calls(prompt_for(arm,contract,evidence,(out/'seed/algorithm.py').read_text(),index),
                    cwd=workspace,writable=False,timeout_s=180)
                if response.returncode: raise RuntimeError(f'Context exited {response.returncode}')
                plan = parse_plan(response.stdout,evidence)
                save(directory/'plan.json',plan)
            chosen = selected_plan(plan)
            stage = 'proposal'
            if not (directory/'program').exists():
                proposal_prompt = contract+'\nImplement this selected research plan faithfully:\n'+canonical(chosen)+'''
Use only NumPy and standard library. Fit cap 30 seconds, 256 training paths.
Do not execute candidate code here, inspect other directories, or use network.
Implement fit/predict CLI exactly. Training payoffs are ALREADY discounted;
do not discount again. Create output directories. Prediction must use current
time_index and causal inputs. Save numerical training diagnostics in the model
directory so the actual learning mechanism can later be inspected. Any fallback
must be reported. Keep implementation compact and runnable.
'''
                original = proposal_agent.run_agent
                proposal_agent.run_agent = calls
                try:
                    ok, detail = proposal_agent.propose(out/'seed',directory/'draft',proposal_prompt,
                        ['algorithm.py','manifest.json'],timeout_s=300,backend='codex',model=model,
                        candidate_mode='python_program',entrypoint='algorithm.py',source_files=['algorithm.py','manifest.json'],
                        intervention=dict(intervention_operator='whole_program_restart',intervention_scope='whole_program',
                            mechanism=chosen['mechanism'],prediction=chosen['prediction'],falsifier=chosen['falsifier']),
                        execution_metadata=metadata)
                finally:
                    proposal_agent.run_agent = original
                if not ok: raise RuntimeError(detail)
                snapshot_source_tree(directory/'draft',directory/'program',1024*1024)
                save(directory/'operator.json',metadata)
            else:
                metadata = pilot.load(directory/'operator.json')
            stage = 'evaluation'
            result = evaluate_once(directory/'program',pilot.request(seed,secret=pilot.run_secret(out)),directory/'public.json')
        except ResourceUnavailable:
            raise
        except Exception as exc:
            result = dict(status='error',score=None,metrics={},evidence={},source_digest=None,
                          failure=f'{type(exc).__name__}: {exc}'[-4000:])
        record = pilot.public_record(f'{arm}-r{index:02d}',result,chosen['mechanism'] if chosen else 'no valid plan',
                                     chosen['structural_difference'] if chosen else None)
        record.update(model=model,seed=seed,arm=arm,round=index,failure_stage=stage if result['status']!='ok' else None,
                      plan_digest=digest(plan),packet_digest=digest(evidence),operator_execution=metadata,
                      protocol_digest=digest(protocol),previous_record_digest=digest(history[-1]),
                      method_transition='not_audited',parent_id='ridge-seed')
        save(directory/'record.json',record)
        save(directory/'artifact_hashes.json',{str(p.relative_to(directory)):pilot.sha(p) for p in directory.rglob('*')
            if p.is_file() and not p.is_symlink() and p.name!='worker.lock'})

def jobs_for_round(index):
    # Paired seeds, counterbalanced arms; time-block model order is randomized.
    jobs=[]
    rng=random.Random(20260906+index)
    for seed in SEEDS:
        models=list(MODELS)
        rng.shuffle(models)
        for offset in range(4):
            for model in models:
                arm=ARMS[(offset+seed+index)%4]
                jobs.append((model,seed,arm,index))
    return jobs

def run_jobs(out,jobs,workers):
    active=[]
    cursor=0
    failures=[]
    while cursor<len(jobs) or active:
        while not failures and cursor<len(jobs) and len(active)<workers:
            model,seed,arm,index=jobs[cursor]
            cursor+=1
            directory=out/'runs'/model/f's{seed}'/arm/f'r{index:02d}'
            if (directory/'record.json').exists(): continue
            directory.mkdir(parents=True,exist_ok=True)
            log=(directory/f'worker-{time.time_ns()}.log').open('x')
            command=[sys.executable,str(Path(__file__).resolve()),'worker','--output',str(out),
                     '--model',model,'--seed',str(seed),'--arm',arm,'--index',str(index)]
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            active.append((proc,log,(model,seed,arm,index)))
        remaining=[]
        for proc,log,job in active:
            rc=proc.poll()
            if rc is None: remaining.append((proc,log,job))
            else:
                log.close()
                if rc: failures.append(dict(job=job,returncode=rc))
        active=remaining
        if failures and not active: break
        time.sleep(1)
    if failures:
        save(out/f'pause-{time.time_ns()}.json',dict(failures=failures,reason='Worker/infrastructure failure. No scientific rerolls; inspect logs before resuming.'))
        raise ResourceUnavailable('Batch paused: '+str(failures))

def supervise(out,workers):
    refuse_unsalted_resume(out)
    verify(out)
    with (out/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(out/f'supervisor-start-{time.time_ns()}.json',dict(pid=os.getpid(),workers=workers))
        secret=pilot.run_secret(out)  # created here, before any worker starts
        for seed in SEEDS:
            path=out/'baselines'/f's{seed}-record.json'
            if path.exists(): continue
            result=evaluate_once(out/'seed',pilot.request(seed,secret=secret),out/'baselines'/f's{seed}-public.json')
            if result['status']!='ok': raise ResourceUnavailable('Baseline preflight failed')
            save(path,pilot.public_record('ridge-seed',result,'Ridge backward continuation regression'))
        for index in range(ROUNDS):
            verify(out)
            run_jobs(out,jobs_for_round(index),workers)
            checkpoint=out/f'round-{index:02d}-complete.json'
            if not checkpoint.exists():
                records=[pilot.load(out/'runs'/m/f's{s}'/a/f'r{i:02d}'/'record.json') for m,s,a,i in jobs_for_round(index)]
                save(checkpoint,dict(round=index,records_digest=digest(records),attempts=len(records),
                    evaluated=sum(r['status']=='ok' for r in records)))
            print(f'ROUND {index+1}/{ROUNDS} COMPLETE',flush=True)
        if not (out/'public_complete.json').exists():
            save(out/'public_complete.json',dict(attempts=len(MODELS)*len(SEEDS)*len(ARMS)*ROUNDS,
                                               frozen_before_private=True))
        # All valid programs get held-out validation, not only handpicked winners.
        for seed in SEEDS:
            evaluate_once(out/'seed',pilot.request(seed,True,secret),out/'private'/f's{seed}-baseline.json')
        for index in range(ROUNDS):
            for model,seed,arm,_ in jobs_for_round(index):
                verify(out)
                directory=out/'runs'/model/f's{seed}'/arm/f'r{index:02d}'
                if pilot.load(directory/'record.json')['status']=='ok':
                    evaluate_once(directory/'program',pilot.request(seed,True,secret),directory/'private.json')
        save(out/'complete.json',dict(public_attempts=3200,mechanism_audit_required=True))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['worker','supervise'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--index',type=int)
    args=parser.parse_args()
    try:
        if args.mode=='worker': attempt(args.output.resolve(),args.model,args.seed,args.arm,args.index)
        else: supervise(args.output.resolve(),args.workers)
    except ResourceUnavailable as exc:
        print(str(exc),file=sys.stderr,flush=True)
        return 75
    return 0

if __name__=='__main__':
    sys.exit(main())
