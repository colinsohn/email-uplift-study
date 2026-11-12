from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from email_uplift_study.cli import render_markdown_summary
from email_uplift_study.pipeline import (
    EXPECTED_SHA256,
    FORBIDDEN_FEATURE_COLUMNS,
    PRE_CAMPAIGN_FEATURES,
    VALID_ACTIONS,
    compare_policies_paired,
    compute_file_sha256,
    evaluate_policy_ipw,
    fit_and_freeze_policies,
    load_dataset,
    make_splits,
    run_duckdb_audit,
    validate_feature_columns,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "data" / "hillstrom-email-experiment.csv"
SQL_PATH = REPO_ROOT / "sql" / "experiment_audit.sql"
RESULTS_JSON_PATH = REPO_ROOT / "results" / "study_results.json"
REPORT_MD_PATH = REPO_ROOT / "results" / "study_report.md"


def _build_synthetic_hillstrom_csv(path: Path, n_rows: int = 6000, seed: int = 42) -> None:
    """Create a deterministic Hillstrom-schema CSV when the gitignored raw CSV is absent."""
    rng = np.random.default_rng(seed)
    segments = np.tile(["Mens E-Mail", "Womens E-Mail", "No E-Mail"], n_rows // 3)
    rng.shuffle(segments)
    recency = rng.integers(1, 13, size=n_rows)
    history = np.round(rng.uniform(29.99, 600.0, size=n_rows), 2)
    hist_seg = np.where(history < 100, "1) $0 - $100", np.where(history < 200, "2) $100 - $200", "3) $200 - $350"))
    mens = rng.integers(0, 2, size=n_rows)
    womens = np.where(mens == 0, 1, rng.integers(0, 2, size=n_rows))
    zip_code = rng.choice(["Urban", "Surburban", "Rural"], size=n_rows)
    newbie = rng.integers(0, 2, size=n_rows)
    channel = rng.choice(["Web", "Phone", "Multichannel"], size=n_rows)

    base_p = 0.10 + 0.03 * (recency <= 4)
    lift = np.where(segments == "Mens E-Mail", 0.08, np.where(segments == "Womens E-Mail", 0.05, 0.0))
    visit = rng.binomial(1, np.clip(base_p + lift, 0.01, 0.95))
    conversion = visit * rng.binomial(1, 0.08, size=n_rows)
    spend = np.round(conversion * rng.uniform(30.0, 150.0, size=n_rows), 2)

    df = pd.DataFrame(
        {
            "recency": recency,
            "history_segment": hist_seg,
            "history": history,
            "mens": mens,
            "womens": womens,
            "zip_code": zip_code,
            "newbie": newbie,
            "channel": channel,
            "segment": segments,
            "visit": visit,
            "conversion": conversion,
            "spend": spend,
        }
    )
    df.to_csv(path, index=False)


def _get_dataset_for_test(tmp_path: Path) -> tuple[pd.DataFrame, Path, bool]:
    if CSV_PATH.exists():
        return load_dataset(CSV_PATH, verify_checksum=True), CSV_PATH, True
    syn_path = tmp_path / "synthetic_hillstrom.csv"
    _build_synthetic_hillstrom_csv(syn_path)
    return load_dataset(syn_path, verify_checksum=False), syn_path, False


def test_split_leakage_and_feature_firewall(tmp_path: Path) -> None:
    # 1. Feature firewall rejects campaign assignment and post-campaign outcomes
    validate_feature_columns(PRE_CAMPAIGN_FEATURES)
    for forbidden in FORBIDDEN_FEATURE_COLUMNS:
        with pytest.raises(ValueError, match="Forbidden column"):
            validate_feature_columns([*PRE_CAMPAIGN_FEATURES, forbidden])

    # 2. Disjoint splits with zero overlap and full row coverage
    df, _, is_real = _get_dataset_for_test(tmp_path)
    train_df, val_df, test_df = make_splits(df, seed=42)

    train_ids = set(train_df["row_id"].tolist())
    val_ids = set(val_df["row_id"].tolist())
    test_ids = set(test_df["row_id"].tolist())

    if is_real:
        assert len(train_ids) == 32000
        assert len(val_ids) == 16000
        assert len(test_ids) == 16000
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids | val_ids | test_ids) == len(df)

    # 3. Verify that test_df cannot leak into policy fitting/tuning
    sub_train = train_df.iloc[:3000].copy()
    sub_val = val_df.iloc[:1500].copy()
    sub_test_clean = test_df.iloc[:1500].copy()

    frozen_a = fit_and_freeze_policies(sub_train, sub_val, budget=0.25, seed=42)
    frozen_b = fit_and_freeze_policies(sub_train.copy(), sub_val.copy(), budget=0.25, seed=42)

    assert frozen_a.regularized_policy.best_c == frozen_b.regularized_policy.best_c
    assert frozen_a.flexible_policy.best_params == frozen_b.flexible_policy.best_params
    assert frozen_a.recency_policy.email_mode == frozen_b.recency_policy.email_mode

    # Predictions on clean test features must be identical and never depend on test outcomes
    corrupted_test = sub_test_clean.copy()
    corrupted_test["visit"] = 1 - corrupted_test["visit"]
    corrupted_test["conversion"] = 1
    corrupted_test["spend"] = 9999.0
    corrupted_test["segment"] = "No E-Mail"

    for name in ("random_targeting", "simple_recency_rule", "regularized_logistic", "flexible_hist_gbdt"):
        acts_clean = frozen_a.predict_actions(name, sub_test_clean, budget=0.25)
        acts_corrupted = frozen_a.predict_actions(name, corrupted_test, budget=0.25)
        assert np.array_equal(acts_clean, acts_corrupted)


def test_contact_budget_enforcement(tmp_path: Path) -> None:
    df, _, _ = _get_dataset_for_test(tmp_path)
    train_df, val_df, test_df = make_splits(df, seed=42)
    frozen = fit_and_freeze_policies(
        train_df.iloc[:3000], val_df.iloc[:1500], budget=0.25, seed=42
    )
    eval_df = test_df.iloc[:1500].copy()
    n = len(eval_df)

    for budget in (0.0, 0.10, 0.25, 0.50):
        expected_contacts = int(np.floor(budget * n))
        for policy_name in (
            "random_targeting",
            "simple_recency_rule",
            "regularized_logistic",
            "flexible_hist_gbdt",
        ):
            actions = frozen.predict_actions(policy_name, eval_df, budget=budget)
            assert len(actions) == n
            assert set(np.unique(actions)).issubset(set(VALID_ACTIONS))
            contacted = int(np.sum(actions != "No E-Mail"))
            assert contacted == expected_contacts, (
                f"{policy_name} at budget={budget} contacted {contacted} != {expected_contacts}"
            )


def test_ipw_estimator_on_toy_experiment_with_known_effect() -> None:
    rows = []
    for stratum in range(4):
        for arm in ("No E-Mail", "Mens E-Mail", "Womens E-Mail"):
            true_prob = 0.50 if (stratum == 0 and arm == "Mens E-Mail") else 0.10
            n_cell = 30
            n_ones = int(round(true_prob * n_cell))
            for idx in range(n_cell):
                rows.append(
                    {
                        "stratum": stratum,
                        "segment": arm,
                        "visit": 1 if idx < n_ones else 0,
                    }
                )
    toy_df = pd.DataFrame(rows)
    policy_actions = np.where(toy_df["stratum"].to_numpy() == 0, "Mens E-Mail", "No E-Mail")
    no_email_actions = np.full(len(toy_df), "No E-Mail", dtype=object)

    est_pol = evaluate_policy_ipw(toy_df, policy_actions, outcome="visit", scale=1000.0)
    est_ctrl = evaluate_policy_ipw(toy_df, no_email_actions, outcome="visit", scale=1000.0)
    paired = compare_policies_paired(
        toy_df, policy_actions, no_email_actions, outcome="visit", scale=1000.0
    )

    assert np.isclose(est_pol["estimate"], 200.0, atol=1e-9)
    assert np.isclose(est_ctrl["estimate"], 100.0, atol=1e-9)
    assert np.isclose(paired["diff"], 100.0, atol=1e-9)
    assert paired["ci95_low"] > 0.0

    rng = np.random.default_rng(12345)
    n_sim = 12000
    asym_probs = {"No E-Mail": 0.50, "Mens E-Mail": 0.25, "Womens E-Mail": 0.25}
    is_target = np.zeros(n_sim, dtype=int)
    is_target[: int(0.25 * n_sim)] = 1

    arms = rng.choice(
        ["No E-Mail", "Mens E-Mail", "Womens E-Mail"],
        size=n_sim,
        p=[asym_probs["No E-Mail"], asym_probs["Mens E-Mail"], asym_probs["Womens E-Mail"]],
    )
    true_p = np.where((is_target == 1) & (arms == "Mens E-Mail"), 0.44, 0.12)
    visits = rng.binomial(1, true_p)
    sim_df = pd.DataFrame({"segment": arms, "visit": visits})

    sim_policy = np.where(is_target == 1, "Mens E-Mail", "No E-Mail")
    sim_no_email = np.full(n_sim, "No E-Mail", dtype=object)

    sim_comp = compare_policies_paired(
        sim_df,
        sim_policy,
        sim_no_email,
        outcome="visit",
        assignment_probabilities=asym_probs,
        scale=1000.0,
    )
    assert sim_comp["ci95_low"] <= 80.0 <= sim_comp["ci95_high"]
    assert abs(sim_comp["diff"] - 80.0) < 15.0


def test_duckdb_audit_and_dataset_checksum(tmp_path: Path) -> None:
    df, csv_file, is_real = _get_dataset_for_test(tmp_path)
    if is_real:
        assert compute_file_sha256(csv_file) == EXPECTED_SHA256
    audit = run_duckdb_audit(csv_file, SQL_PATH)

    assert audit["missing_values"]["total_rows"] == len(df)
    assert audit["missing_values"]["invalid_range_rows"] == 0
    for key, val in audit["missing_values"].items():
        if key.startswith("null_"):
            assert val == 0

    assert len(audit["arm_sizes"]) == 3
    for row in audit["pre_campaign_balance"]:
        assert abs(row["smd_mens_vs_control"]) < 0.08
        assert abs(row["smd_womens_vs_control"]) < 0.08

    ate_by_arm = {r["comparison_arm"]: r for r in audit["experiment_ate_summary"]}
    assert ate_by_arm["Mens E-Mail"]["ate_visits_ci95_low"] > 20.0
    assert ate_by_arm["Womens E-Mail"]["ate_visits_ci95_low"] > 10.0


def test_report_generator_matches_study_results_json() -> None:
    results = json.loads(RESULTS_JSON_PATH.read_text(encoding="utf-8"))
    rendered = render_markdown_summary(results)
    saved_report = REPORT_MD_PATH.read_text(encoding="utf-8")

    assert rendered == saved_report
    assert "+14.62" in rendered
    assert "+21.00" in rendered
    assert "29.4%" in rendered
    assert "+12.38" not in rendered
    assert "30.6%" not in rendered
