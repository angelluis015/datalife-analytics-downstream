import math


def compare_metric(
    baseline_metric: float,
    current_metric: float,
    tolerance: float,
) -> bool:
    for value in (baseline_metric, current_metric, tolerance):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Metric comparison requires numeric inputs")

        try:
            if not math.isfinite(value):
                raise ValueError("Metric comparison requires finite values")
        except OverflowError as exc:
            raise ValueError("Metric comparison value is too large") from exc

    if tolerance < 0:
        raise ValueError("Tolerance cannot be negative")

    try:
        difference = abs(current_metric - baseline_metric)
    except OverflowError as exc:
        raise ValueError("Metric difference is too large") from exc

    if not math.isfinite(difference):
        raise ValueError("Metric difference must be finite")

    return difference <= tolerance


def get_metric(report: dict, metric: str, mode: str) -> float:
    test = report["models"]["ridge_logistic"]["test"]

    if metric == "roc_auc" and mode == "kfold":
        return test["roc_auc_foldwise"]["mean"]

    return test[metric]


def get_split_mode(report: dict) -> str:
    split = report.get("split")

    if not isinstance(split, dict):
        raise ValueError("Invalid split section")

    mode = split.get("mode")

    if mode == "subject_stratified_kfold":
        return "kfold"

    if mode is None and "n_train" in split and "n_test" in split:
        return "single"

    raise ValueError("Unknown split mode")


def compare_exact_fields(
    baseline_section: dict,
    current_section: dict,
    fields: list[str],
    prefix: str,
    checks: dict,
) -> None:
    for field in fields:
        name = f"{prefix}.{field}"

        if field not in baseline_section or field not in current_section:
            checks[name] = "MISSING"
        elif baseline_section[field] == current_section[field]:
            checks[name] = "PASS"
        else:
            checks[name] = "FAIL"


def compare_reports(baseline_report: dict, current_report: dict, tolerances: dict) -> dict:

    lineage_fields = [
        "contract",
        "contract_version",
        "generator_seed",
        "split_seed",
        "train_fraction",
        "n_subjects",
    ]

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

    baseline_mode = get_split_mode(baseline_report)
    current_mode = get_split_mode(current_report)

    if baseline_mode != current_mode:
        raise ValueError("Cannot compare different split modes")

    overall_result = "PASS"
    checks = {}

    for key, value in tolerances.items():
        try:
            result = compare_metric(
                get_metric(baseline_report, key, baseline_mode),
                get_metric(current_report, key, current_mode),
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

    compare_exact_fields(
        baseline_report["lineage"],
        current_report["lineage"],
        lineage_fields,
        "lineage",
        checks,
    )

    split_fields = (
        ["n_train", "n_test", "train_subjects", "test_subjects"]
        if baseline_mode == "single"
        else ["mode", "n_splits", "n_examples"]
    )

    compare_exact_fields(
        baseline_report["split"],
        current_report["split"],
        split_fields,
        "split",
        checks,
    )

    compare_exact_fields(
        baseline_report["models"]["ridge_logistic"]["test"],
        current_report["models"]["ridge_logistic"]["test"],
        ["n"],
        "test",
        checks,
    )
    if any(status != "PASS" for status in checks.values()):
        overall_result = "FAIL"

    return {"status": overall_result, "checks": checks}
