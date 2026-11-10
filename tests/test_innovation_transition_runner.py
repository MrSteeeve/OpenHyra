import importlib.util
import json
from pathlib import Path


def runner():
    path=Path(__file__).resolve().parents[1]/"experiments/run_innovation_transition.py"
    spec=importlib.util.spec_from_file_location("transition_runner_test",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_backend_launch_failure_is_retained(tmp_path,monkeypatch):
    module=runner()
    def fail(*args,**kwargs):
        raise OSError("missing executable")
    monkeypatch.setattr(module,"run_agent",fail)
    calls=module.Calls(tmp_path,"test")
    result=calls("prompt",timeout_s=1,writable=False)
    assert result.returncode==125
    record=json.loads((tmp_path/"00-output.json").read_text())
    assert record["returncode"]==125 and "missing executable" in record["stderr"]


def test_contract_is_frozen_not_only_python():
    hashes=runner().frozen_files()
    assert "tasks/bermudan_python_search/TASK.md" in hashes
    assert "tasks/bermudan_python_search/research_candidates/linear_ridge/manifest.json" in hashes


def test_one_private_salt_and_audit_seed_per_run(tmp_path):
    module=runner()
    secret=module.run_secret(tmp_path)
    assert module.run_secret(tmp_path)==secret
    public,private=module.request(3,secret=secret),module.request(3,True,secret)
    assert public["config"]["pricing_salt"]==private["config"]["pricing_salt"]==secret["pricing_salt"]
    assert private["seed"]==secret["audit_seed_offset"]+3
    template=module.request(3,True)
    assert template["seed"] is None and "pricing_salt" not in template["config"]


def test_saved_requests_omit_the_salt(tmp_path,monkeypatch):
    module=runner()
    secret=module.run_secret(tmp_path)
    source=tmp_path/"src"; source.mkdir(); (source/"algorithm.py").write_text("")
    seen={}
    def fake_run(command,**kwargs):
        seen["request"]=json.loads(Path(command[-1]).read_text())
        return module.subprocess.CompletedProcess(command,0,json.dumps(dict(score=0.0,metrics={},evidence={})),"")
    monkeypatch.setattr(module.subprocess,"run",fake_run)
    value=module.evaluate(source,module.request(3,secret=secret),tmp_path/"out"/"public.json")
    assert seen["request"]["config"]["pricing_salt"]==secret["pricing_salt"]
    saved=[p.read_text() for p in (tmp_path/"out").glob("*.json")]
    assert len(saved)==2 and not any(str(secret["pricing_salt"]) in text for text in saved)
    assert "pricing_salt" not in value["request"]["config"]
