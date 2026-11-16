from datalife_analytics.baseline.diff import compare_reports


def make_report(accuracy: float = 0.80) -> dict:
    return {
        "lineage": {
            "contract": "synthetic-contract",
            "contract_version": "0.1.0",
            "generator_seed": 42,
        },
        "models": {
            "ridge_logistic": {
                "test": {
                    "accuracy": accuracy,
                    "brier": 0.15,
                    "roc_auc": 0.85,
                }
            }
        },
    }

def test_identical_reports_pass():
    baseline = make_report()
    current = make_report()

    tolerances = {"accuracy": 0.02}

    result = compare_reports(baseline, current, tolerances)

    assert result["status"] == "PASS"
    assert result["checks"]["accuracy"] == "PASS"


def test_metric_exceeding_tolerance_fails():
    baseline = make_report(accuracy=0.80)
    current = make_report(accuracy=0.90)

    tolerances = {"accuracy": 0.02}

    result = compare_reports(baseline, current, tolerances)

    assert result["status"] == "FAIL"
    assert result["checks"]["accuracy"] == "FAIL"

def test_missing_metric_fails():
    baseline = make_report()
    current = make_report()

    del current["models"]["ridge_logistic"]["test"]["accuracy"]

    tolerances = {"accuracy": 0.02}

    result = compare_reports(baseline, current, tolerances)

    assert result["status"] == "FAIL"
    assert result["checks"]["accuracy"] == "MISSING"

def test_contract_version_mismatch_fails():
    baseline = make_report()
    current = make_report()

    current["lineage"]["contract_version"] = "0.2.0"

    tolerances = {"accuracy": 0.02}

    result = compare_reports(baseline, current, tolerances)

    assert result["status"] == "FAIL"
    assert result["checks"]["contract_version"] == "FAIL"
