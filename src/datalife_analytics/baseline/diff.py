import math


def compare_metric(baseline_metric: float, current_metric: float, tolerance: float) -> bool:
    for value in (baseline_metric, current_metric, tolerance):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Metric comparison requires numeric inputs")

        if not math.isfinite(value):
            raise ValueError("Metric comparison requires finite values")

    if tolerance < 0:
        raise ValueError("Tolerance cannot be negative")

    return abs(current_metric - baseline_metric) <= tolerance


def compare_reports(baseline_report: dict, current_report: dict, tolerances: dict) -> dict:
    lineage_fields = ["contract", "contract_version", "generator_seed"]

    if any(key in lineage_fields for key in tolerances):
        raise ValueError("Tolerance keys cannot overlap with lineage fields")

    for report in (baseline_report, current_report):
        if not isinstance(report.get("lineage"), dict):
            raise ValueError("Invalid lineage section")

        models = report.get("models")
        if not isinstance(models, dict):
            raise ValueError("Invalid models section")

        model = models.get("ridge_logistic")
        if not isinstance(model, dict):
            raise ValueError("Invalid model section")

        if not isinstance(model.get("test"), dict):
            raise ValueError("Invalid test metrics section")

    overall_result = "PASS"
    checks = {}

    for key, value in tolerances.items():
        try:
            result = compare_metric(
                baseline_report["models"]["ridge_logistic"]["test"][key],
                current_report["models"]["ridge_logistic"]["test"][key],
                value,
            )
            if result:
                checks[key] = "PASS"
            else:
                checks[key] = "FAIL"
                overall_result = "FAIL"
        except KeyError:
            checks[key] = "MISSING"
            overall_result = "FAIL"

    for key in lineage_fields:
        try:
            lineage_check = baseline_report["lineage"][key] == current_report["lineage"][key]
            if lineage_check:
                checks[key] = "PASS"
            else:
                checks[key] = "FAIL"
                overall_result = "FAIL"
        except KeyError:
            checks[key] = "MISSING"
            overall_result = "FAIL"
    return {"status": overall_result, "checks": checks}
