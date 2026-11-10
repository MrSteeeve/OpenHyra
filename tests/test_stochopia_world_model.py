from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stochopia.research.forecast_protocol")
pytest.importorskip("structurerdsl")

from stochopia.research import forecast_protocol as fp  # noqa: E402
from stochopia.research.market_panel import MarketPanel  # noqa: E402

from tasks.stochopia_world_model import generator as gen  # noqa: E402
from tasks.stochopia_world_model.evaluator import (  # noqa: E402
    SEED_PROGRAM,
    evaluate_program,
    public_feedback,
)

needs_seatbelt = pytest.mark.skipif(sys.platform != "darwin", reason="training sandbox requires macOS Seatbelt")


def _panel(sessions: int = 1000) -> MarketPanel:
    rng = np.random.default_rng(7)
    days, current = [], date(2020, 1, 1)
    while len(days) < sessions:
        if current.weekday() < 5:
            days.append(current.isoformat())
        current += timedelta(days=1)
    close = 5000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, size=(sessions, 2)), axis=0))
    iv = 0.2 + 0.01 * rng.standard_normal(sessions)
    carry = 0.1 + 0.01 * rng.standard_normal((sessions, 2))
    return MarketPanel(tuple(days), close, np.full(sessions, 0.02), carry, iv, {"schema": "test"})


@needs_seatbelt
def test_seed_program_reproduces_reference_exactly(tmp_path):
    panel = _panel()
    outcome = evaluate_program(SEED_PROGRAM.read_text(), panel=panel, split="dev", workdir=tmp_path)
    assert outcome["status"] == "ok"
    assert outcome["score"] == pytest.approx(0.0, abs=1e-12)
    assert outcome["cost"]["origins_run"] == len(fp.origin_indices(panel, "dev"))
    assert list(tmp_path.iterdir()) == []  # sandbox cells are cleaned up
    feedback = public_feedback(outcome["report"])
    assert "dates" not in json.dumps(feedback)


_WRITE_VALID = (
    "import numpy as np, sys\nfrom pathlib import Path\n"
    "out = Path(sys.argv[sys.argv.index('--output') + 1])\n"
    "np.savez(out / 'forecast.npz', returns=np.zeros((256, 21, 2)), q=np.full((256, 3), 0.2))\n"
)


@needs_seatbelt
@pytest.mark.parametrize(
    "program, status, needle",
    [
        ("def broken(:\n", "invalid_program", "SyntaxError"),
        # Each escape attempt writes a valid forecast if it succeeds, so only a
        # denial by the sandbox can produce forecast_error.
        ("open(SECRET).read()\n" + _WRITE_VALID, "forecast_error", "PermissionError"),
        ("open(SECRET.replace('secret.txt', 'leak.txt'), 'w').write('x')\n" + _WRITE_VALID,
         "forecast_error", "PermissionError"),
        ("import numpy as np, sys\nfrom pathlib import Path\n"
         "out = Path(sys.argv[sys.argv.index('--output') + 1])\n"
         "(out / 'forecast.npz').write_bytes(b'PK\\x03\\x04garbage')\n", "forecast_error", "unreadable"),
        ("import numpy as np, sys\nfrom pathlib import Path\n"
         "out = Path(sys.argv[sys.argv.index('--output') + 1])\n"
         "np.savez(out / 'forecast.npz', returns=np.full((256, 21, 2), 'x'), q=np.ones((256, 3)))\n",
         "forecast_error", "numeric"),
        ("import numpy as np, sys\nfrom pathlib import Path\n"
         "out = Path(sys.argv[sys.argv.index('--output') + 1])\n"
         "np.savez(out / 'forecast.npz', returns=np.zeros((2, 2, 2)), q=np.ones((2, 3)))\n",
         "forecast_error", "shape"),
    ],
)
def test_invalid_or_escaping_programs_are_rejected(tmp_path, program, status, needle):
    secret = tmp_path / "secret.txt"
    secret.write_text("do not read")
    program = program.replace("SECRET", repr(str(secret.resolve())))
    outcome = evaluate_program(program, panel=_panel(), split="dev", workdir=tmp_path / "work")
    assert outcome["status"] == status
    assert outcome["score"] is None
    assert needle in outcome["error"]
    assert not (tmp_path / "leak.txt").exists()
    assert str(Path.home()) not in outcome["error"]


def test_parse_program_response_requires_both_blocks():
    text = (
        "```json\n{\"mechanism\": \"m\", \"predicted_improvements\": [\"volatility\"], \"falsifier\": \"f\"}\n```\n"
        "```python\nprint('x')\n```"
    )
    parsed = gen.parse_program_response(text)
    assert parsed["program"].strip() == "print('x')"
    assert parsed["predicted_improvements"] == ["volatility"]
    with pytest.raises(ValueError, match="python"):
        gen.parse_program_response("```json\n{}\n```")
    nested = (
        "```json\n{\"mechanism\": \"m\"}\n```\n"
        "```py\ndef f():\n    \"\"\"Example: ```x``` inline.\"\"\"\n    return 1\n```\n"
    )
    assert "return 1" in gen.parse_program_response(nested)["program"]


def test_api_generator_sends_one_request_and_parses(monkeypatch):
    httpx = pytest.importorskip("httpx")
    seen = {}

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            content = ("```json\n{\"mechanism\": \"garch\", \"predicted_improvements\": [\"volatility\"], "
                       "\"falsifier\": \"no gain\"}\n```\n```python\nprint(1)\n```")
            return {"model": "m", "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    def post(url, json, headers, timeout):
        seen.update(url=url, body=json, headers=headers)
        return Response()

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setenv("TEST_WM_KEY", "secret")
    generator = gen.APIProgramGenerator(base_url="https://example.test/v1", model="m",
                                        api_key_env="TEST_WM_KEY", max_calls=1)
    result = generator.generate({"round": 0})
    assert result.error is None
    assert result.proposal["mechanism"] == "garch"
    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["body"]["messages"][0]["content"] == gen.SYSTEM_PROMPT
    assert "secret" not in json.dumps(generator.metadata())
    with pytest.raises(RuntimeError, match="budget"):
        generator.generate({"round": 1})


def test_system_prompt_contains_no_calendar_years():
    import re

    assert not re.search(r"\b(19|20)\d{2}\b", gen.SYSTEM_PROMPT)


@needs_seatbelt
def test_offline_loop_runs_end_to_end(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import run_stochopia_world_model as runner

    summary = runner.run(tmp_path / "out", gen.OfflineProgramGenerator(), panel=_panel(), rounds=1, per_round=2)
    assert summary["candidates"] == 2
    assert summary["evaluated_ok"] == 2
    assert summary["test_audit"]["selected"]["status"] == "ok"
    experience = [json.loads(line) for line in (tmp_path / "out" / "experience.jsonl").read_text().splitlines()]
    assert all(item["parent_id"] == "seed" for item in experience)
    assert all(item["prediction_check"]["hit_rate"] is not None for item in experience)
    ledger = (tmp_path / "out" / "ledger.jsonl").read_text().splitlines()
    assert len(ledger) == 2
