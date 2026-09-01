import json

from probe_suite import ProbeSuite, ProbeResult, ProbeReport


def test_add_and_run_passing_probe(tmp_path):
    suite = ProbeSuite(tmp_path)
    suite.add_probe("passing", {"answer": 42}, 1.0)

    report = suite.run(lambda solution: (1.0, "ok", "", {}))

    assert len(report.results) == 1
    assert report.results[0].passed is True
    assert report.results[0].actual == 1.0


def test_run_failing_score_mismatch(tmp_path):
    suite = ProbeSuite(tmp_path)
    suite.add_probe("mismatch", {}, 1.0)

    report = suite.run(lambda solution: (0.5, "ok", "", {}))

    assert report.results[0].passed is False
    assert "score mismatch" in report.results[0].error


def test_run_evaluator_crash(tmp_path):
    suite = ProbeSuite(tmp_path)
    suite.add_probe("crash", {}, 1.0)

    def crashing_evaluator(solution):
        raise RuntimeError("boom")

    report = suite.run(crashing_evaluator)

    assert report.results[0].passed is False
    assert "RuntimeError('boom')" in report.results[0].error


def test_run_evaluator_bad_status(tmp_path):
    suite = ProbeSuite(tmp_path)
    suite.add_probe("bad-status", {}, 1.0)

    report = suite.run(lambda solution: (1.0, "crash", "", {}))

    assert report.results[0].passed is False
    assert "status=crash" in report.results[0].error


def test_run_evaluator_none_score(tmp_path):
    suite = ProbeSuite(tmp_path)
    suite.add_probe("none-score", {}, 1.0)

    report = suite.run(lambda solution: (None, "ok", "", {}))

    assert report.results[0].passed is False
    assert report.results[0].actual is None
    assert "None score" in report.results[0].error


def test_all_passed_true_when_all_pass(tmp_path):
    suite = ProbeSuite(tmp_path)
    suite.add_probe("first", {"score": 1.0}, 1.0)
    suite.add_probe("second", {"score": 2.0}, 2.0)

    report = suite.run(
        lambda solution: (solution["score"], "ok", "", {})
    )

    assert report.all_passed is True
    assert all(result.passed for result in report.results)


def test_all_passed_false_when_any_fails(tmp_path):
    suite = ProbeSuite(tmp_path)
    suite.add_probe("passing", {"score": 1.0}, 1.0)
    suite.add_probe("failing", {"score": 0.0}, 1.0)

    report = suite.run(
        lambda solution: (solution["score"], "ok", "", {})
    )

    assert report.all_passed is False
    assert [result.passed for result in report.results] == [True, False]


def test_save_and_load_report(tmp_path):
    suite = ProbeSuite(tmp_path)
    report = ProbeReport(
        suite_id="probe_saved",
        timestamp="2026-09-01T00:00:00+00:00",
        results=[
            ProbeResult(
                probe_id="saved",
                passed=True,
                expected=1.0,
                actual=1.0,
                elapsed_s=0.01,
            )
        ],
        all_passed=True,
    )

    path = suite.save_report(report, tmp_path / "reports")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == "probe_saved.json"
    assert loaded == report.to_dict()
