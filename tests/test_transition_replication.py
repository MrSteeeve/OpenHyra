import importlib.util
import json
from pathlib import Path
import pytest


def module():
    p=Path(__file__).resolve().parents[1]/'experiments/run_transition_replication.py'
    spec=importlib.util.spec_from_file_location('replication_test',p)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_factorial_changes_only_independent_factors():
    m=module()
    prompts={a:m.prompt_for(a,'CONTRACT',{'record_ids':[]},'SECRET_PARENT',0) for a in m.ARMS}
    for arm,text in prompts.items():
        assert ('SECRET_PARENT' in text)==arm.startswith('visible_')
        assert ('Select a nonlocal alternative' in text)==arm.endswith('_explore')
        assert 'exactly 3 objects' in text
    def strip_source(text): return text.split('\nParent implementation')[0]
    assert strip_source(prompts['visible_generic'])==strip_source(prompts['hidden_generic'])
    assert strip_source(prompts['visible_explore'])==strip_source(prompts['hidden_explore'])


def test_history_keeps_all_public_failures_and_excludes_private():
    m=module()
    rows=[dict(id=str(i),data_role='public',status='error',failure='timeout',observations=[]) for i in range(20)]
    result=m.packet(rows+[dict(id='private-leak',data_role='private',score=99)])
    assert result['record_ids']==[str(i) for i in range(20)]
    assert all(r['failure']=='timeout' for r in result['records'])
    assert 'private-leak' not in json.dumps(result)


def test_atomic_exclusive_records(tmp_path):
    m=module()
    m.save(tmp_path/'record.json',{'v':1})
    with pytest.raises(FileExistsError):m.save(tmp_path/'record.json',{'v':2})
    assert json.loads((tmp_path/'record.json').read_text())=={'v':1}
    assert not list(tmp_path.glob('*.tmp-*'))


def test_schedule_exact_paired_design():
    m=module()
    all_jobs=[job for i in range(m.ROUNDS) for job in m.jobs_for_round(i)]
    assert len(all_jobs)==len(set(all_jobs))==3200
    for index in range(20):
        jobs=m.jobs_for_round(index)
        assert len(jobs)==160
        assert jobs==m.jobs_for_round(index)
        for model in m.MODELS:
            for seed in m.SEEDS:
                assert {j[2] for j in jobs if j[:2]==(model,seed)}==set(m.ARMS)


def test_interrupted_call_is_not_silently_retried(tmp_path,monkeypatch):
    m=module()
    m.save(tmp_path/'00-input.json',{'prompt':'old'})
    def fail(*args,**kwargs): raise AssertionError('must not call backend')
    monkeypatch.setattr(m,'_run_cli',fail)
    with pytest.raises(m.ResourceUnavailable):
        m.Calls(tmp_path,'gpt-5.5')('new',cwd=tmp_path,timeout_s=1)


def test_medium_model_binding_and_retained_timeout(tmp_path,monkeypatch):
    m=module()
    def timeout(cmd,**kwargs):
        assert cmd[cmd.index('--model')+1]=='gpt-5.6-terra'
        assert 'model_reasoning_effort="medium"' in cmd
        raise m.subprocess.TimeoutExpired(cmd,1,output=b'partial',stderr=b'timed out')
    monkeypatch.setattr(m,'_run_cli',timeout)
    r=m.Calls(tmp_path,'gpt-5.6-terra')('prompt',cwd=tmp_path,timeout_s=1)
    assert r.returncode==124
    assert json.loads((tmp_path/'00-output.json').read_text())['stdout']=='partial'


def test_quota_failure_pauses_instead_of_becoming_model_failure(tmp_path,monkeypatch):
    m=module()
    monkeypatch.setattr(m,'_run_cli',lambda cmd,**kw:m.subprocess.CompletedProcess(cmd,1,'','Usage limit reached'))
    with pytest.raises(m.ResourceUnavailable):
        m.Calls(tmp_path,'gpt-5.5')('prompt',cwd=tmp_path,timeout_s=1)
    assert json.loads((tmp_path/'00-output.json').read_text())['returncode']==1


def test_pre_salt_output_dir_is_refused_with_one_message(tmp_path):
    m=module()
    (tmp_path/'baselines').mkdir()
    with pytest.raises(SystemExit, match='fresh --output'):
        m.supervise(tmp_path,1)
