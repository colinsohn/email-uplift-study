from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXPECTED_SHA256 = "27bab8c5d3669f26ec08ebb50a0a78317542f29501156f2e2af6781fab4cd7e2"

PRE_CAMPAIGN_FEATURES: tuple[str, ...] = (
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
)
NUMERIC_FEATURES: tuple[str, ...] = ("recency", "history", "mens", "womens", "newbie")
CATEGORICAL_FEATURES: tuple[str, ...] = ("history_segment", "zip_code", "channel")

FORBIDDEN_FEATURE_COLUMNS: tuple[str, ...] = ("segment", "visit", "conversion", "spend")
VALID_ACTIONS: tuple[str, ...] = ("No E-Mail", "Mens E-Mail", "Womens E-Mail")

# Documented randomization probabilities in Hillstrom's 3-arm email experiment.
ASSIGNMENT_PROBABILITIES: dict[str, float] = {
    "No E-Mail": 1.0 / 3.0,
    "Mens E-Mail": 1.0 / 3.0,
    "Womens E-Mail": 1.0 / 3.0,
}

Z_95 = 1.959963984540054


def compute_file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_feature_columns(columns: list[str] | tuple[str, ...]) -> None:
    """Ensure campaign assignment and post-campaign outcomes never enter customer features."""
    forbidden_found = [col for col in columns if col in FORBIDDEN_FEATURE_COLUMNS]
    if forbidden_found:
        raise ValueError(
            f"Forbidden column(s) in customer features: {forbidden_found}. "
            f"Campaign assignment and post-campaign outcomes must be excluded."
        )


def load_dataset(csv_path: Path | str, verify_checksum: bool = True) -> pd.DataFrame:
    path = Path(csv_path)
    if verify_checksum:
        actual = compute_file_sha256(path)
        if actual != EXPECTED_SHA256:
            raise ValueError(f"Checksum mismatch for {path}: expected {EXPECTED_SHA256}, got {actual}")
    df = pd.read_csv(path)
    df["row_id"] = np.arange(len(df), dtype=int)
    return df


def parse_named_sql_queries(sql_path: Path | str) -> dict[str, str]:
    text = Path(sql_path).read_text(encoding="utf-8")
    queries: dict[str, str] = {}
    current_name: str | None = None
    buffer: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("-- @query:"):
            if current_name and buffer:
                queries[current_name] = "\n".join(buffer).strip()
            current_name = stripped.split(":", 1)[1].strip()
            buffer = []
        elif current_name is not None:
            buffer.append(line)
    if current_name and buffer:
        queries[current_name] = "\n".join(buffer).strip()
    return queries


def run_duckdb_audit(csv_path: Path | str, sql_path: Path | str) -> dict[str, Any]:
    """Execute the DuckDB SQL audit queries against the raw CSV."""
    validate_feature_columns(PRE_CAMPAIGN_FEATURES)
    queries = parse_named_sql_queries(sql_path)
    con = duckdb.connect(database=":memory:")
    escaped_csv = str(Path(csv_path)).replace("'", "''")
    con.execute(f"CREATE VIEW hillstrom_raw AS SELECT * FROM read_csv_auto('{escaped_csv}');")

    arm_sizes = con.execute(queries["arm_sizes"]).df().to_dict(orient="records")
    missing_rows = con.execute(queries["missing_values"]).df().to_dict(orient="records")
    balance_rows = con.execute(queries["pre_campaign_balance"]).df().to_dict(orient="records")
    ate_rows = con.execute(queries["experiment_ate_summary"]).df().to_dict(orient="records")
    con.close()

    return {
        "arm_sizes": arm_sizes,
        "missing_values": missing_rows[0],
        "pre_campaign_balance": balance_rows,
        "experiment_ate_summary": ate_rows,
    }


def make_splits(
    df: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create fixed 50% train, 25% validation, 25% test splits stratified by arm and visit."""
    strata = df["segment"].astype(str) + "_" + df["visit"].astype(str)
    train_df, temp_df = train_test_split(
        df, test_size=0.50, random_state=seed, stratify=strata
    )
    temp_strata = temp_df["segment"].astype(str) + "_" + temp_df["visit"].astype(str)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_strata
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def ipw_pseudo_outcomes(
    df_eval: pd.DataFrame,
    actions: np.ndarray,
    outcome: str = "visit",
    assignment_probabilities: dict[str, float] | None = None,
    scale: float = 1000.0,
) -> np.ndarray:
    """Compute individual Horvitz-Thompson IPW pseudo-outcomes from randomized assignment.

    Because treatment A_i was randomized with known probability p(A_i),
    psi_i(pi) = (I(A_i == pi(X_i)) / p(A_i)) * Y_i is strictly unbiased for E[Y_i(pi(X_i))].
    """
    probs = assignment_probabilities or ASSIGNMENT_PROBABILITIES
    observed_arm = df_eval["segment"].to_numpy()
    chosen_arm = np.asarray(actions, dtype=object)
    y = df_eval[outcome].to_numpy(dtype=float)
    arm_prob = np.array([probs[str(a)] for a in observed_arm], dtype=float)
    match = (observed_arm == chosen_arm).astype(float)
    return (match / arm_prob) * y * scale


def evaluate_policy_ipw(
    df_eval: pd.DataFrame,
    actions: np.ndarray,
    outcome: str = "visit",
    assignment_probabilities: dict[str, float] | None = None,
    scale: float = 1000.0,
) -> dict[str, float]:
    scores = ipw_pseudo_outcomes(df_eval, actions, outcome, assignment_probabilities, scale)
    mean_val = float(np.mean(scores))
    se_val = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
    return {
        "estimate": mean_val,
        "se": se_val,
        "ci95_low": mean_val - Z_95 * se_val,
        "ci95_high": mean_val + Z_95 * se_val,
    }


def compare_policies_paired(
    df_eval: pd.DataFrame,
    actions_candidate: np.ndarray,
    actions_reference: np.ndarray,
    outcome: str = "visit",
    assignment_probabilities: dict[str, float] | None = None,
    scale: float = 1000.0,
) -> dict[str, float]:
    """Compute paired IPW difference between two policies on the same held-out customers.

    On every customer where actions_candidate[i] == actions_reference[i], the paired
    pseudo-outcome difference d_i is identically 0, cancelling shared baseline variance.
    """
    s_cand = ipw_pseudo_outcomes(
        df_eval, actions_candidate, outcome, assignment_probabilities, scale
    )
    s_ref = ipw_pseudo_outcomes(
        df_eval, actions_reference, outcome, assignment_probabilities, scale
    )
    diff = s_cand - s_ref
    mean_diff = float(np.mean(diff))
    se_diff = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
    disagree_rate = float(
        np.mean(np.asarray(actions_candidate, dtype=object) != np.asarray(actions_reference, dtype=object))
    )
    return {
        "diff": mean_diff,
        "se": se_diff,
        "ci95_low": mean_diff - Z_95 * se_diff,
        "ci95_high": mean_diff + Z_95 * se_diff,
        "disagreement_rate": disagree_rate,
    }


def select_top_budget_from_scores(
    uplift_scores: np.ndarray,
    best_email_actions: np.ndarray,
    row_ids: np.ndarray,
    budget: float,
) -> np.ndarray:
    """Contact exact floor(budget * N) highest-uplift customers; send No E-Mail to the rest."""
    n = len(uplift_scores)
    k = int(np.floor(budget * n))
    actions = np.full(n, "No E-Mail", dtype=object)
    if k <= 0:
        return actions
    order = np.lexsort((row_ids, -uplift_scores))
    chosen = order[:k]
    actions[chosen] = best_email_actions[chosen]
    return actions


@dataclass(frozen=True)
class RandomTargetingPolicy:
    seed: int = 42

    def predict_actions(self, df: pd.DataFrame, budget: float = 0.25) -> np.ndarray:
        validate_feature_columns(PRE_CAMPAIGN_FEATURES)
        n = len(df)
        k = int(np.floor(budget * n))
        actions = np.full(n, "No E-Mail", dtype=object)
        if k <= 0:
            return actions
        # Deterministic per-row hash derived from pre-campaign row_id and seed
        rng = np.random.default_rng(self.seed)
        scores = rng.random(n)
        email_choices = rng.choice(["Mens E-Mail", "Womens E-Mail"], size=n)
        order = np.lexsort((df["row_id"].to_numpy(), -scores))
        chosen = order[:k]
        actions[chosen] = email_choices[chosen]
        return actions


@dataclass(frozen=True)
class SimpleRecencyPolicy:
    email_mode: str  # 'mens_default' or 'category_affinity', frozen on train+val
    val_tuning_history: list[dict[str, Any]]

    def predict_actions(
        self, df: pd.DataFrame, budget: float = 0.25, mode_override: str | None = None
    ) -> np.ndarray:
        validate_feature_columns(PRE_CAMPAIGN_FEATURES)
        n = len(df)
        k = int(np.floor(budget * n))
        actions = np.full(n, "No E-Mail", dtype=object)
        if k <= 0:
            return actions
        # Rank most recent purchasers first (recency ascending), breaking ties by higher prior history spend
        order = np.lexsort(
            (
                df["row_id"].to_numpy(),
                -df["history"].to_numpy(dtype=float),
                df["recency"].to_numpy(dtype=float),
            )
        )
        chosen = order[:k]
        mode = mode_override or self.email_mode
        if mode == "mens_default":
            actions[chosen] = "Mens E-Mail"
        else:
            womens_only = (df["womens"].to_numpy() == 1) & (df["mens"].to_numpy() == 0)
            email_ver = np.where(womens_only, "Womens E-Mail", "Mens E-Mail")
            actions[chosen] = email_ver[chosen]
        return actions


@dataclass(frozen=True)
class RegularizedActionResponsePolicy:
    preprocessor: ColumnTransformer
    models: dict[str, LogisticRegression]
    best_c: float
    val_tuning_history: list[dict[str, Any]]

    def score_actions(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        validate_feature_columns(PRE_CAMPAIGN_FEATURES)
        X = self.preprocessor.transform(df[list(PRE_CAMPAIGN_FEATURES)])
        return {act: self.models[act].predict_proba(X)[:, 1] for act in VALID_ACTIONS}

    def predict_actions(self, df: pd.DataFrame, budget: float = 0.25) -> np.ndarray:
        probs = self.score_actions(df)
        p_no = probs["No E-Mail"]
        p_mens = probs["Mens E-Mail"]
        p_womens = probs["Womens E-Mail"]
        best_email = np.where(p_mens >= p_womens, "Mens E-Mail", "Womens E-Mail")
        uplift = np.maximum(p_mens, p_womens) - p_no
        return select_top_budget_from_scores(
            uplift, best_email, df["row_id"].to_numpy(), budget
        )


@dataclass(frozen=True)
class FlexibleActionResponsePolicy:
    preprocessor: ColumnTransformer
    models: dict[str, HistGradientBoostingClassifier]
    best_params: dict[str, Any]
    val_tuning_history: list[dict[str, Any]]

    def score_actions(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        validate_feature_columns(PRE_CAMPAIGN_FEATURES)
        X = self.preprocessor.transform(df[list(PRE_CAMPAIGN_FEATURES)])
        return {act: self.models[act].predict_proba(X)[:, 1] for act in VALID_ACTIONS}

    def predict_actions(self, df: pd.DataFrame, budget: float = 0.25) -> np.ndarray:
        probs = self.score_actions(df)
        p_no = probs["No E-Mail"]
        p_mens = probs["Mens E-Mail"]
        p_womens = probs["Womens E-Mail"]
        best_email = np.where(p_mens >= p_womens, "Mens E-Mail", "Womens E-Mail")
        uplift = np.maximum(p_mens, p_womens) - p_no
        return select_top_budget_from_scores(
            uplift, best_email, df["row_id"].to_numpy(), budget
        )


@dataclass(frozen=True)
class FrozenPolicySuite:
    random_policy: RandomTargetingPolicy
    recency_policy: SimpleRecencyPolicy
    regularized_policy: RegularizedActionResponsePolicy
    flexible_policy: FlexibleActionResponsePolicy

    def predict_actions(
        self, policy_name: str, df: pd.DataFrame, budget: float = 0.25
    ) -> np.ndarray:
        if policy_name == "no_email":
            return np.full(len(df), "No E-Mail", dtype=object)
        if policy_name == "random_targeting":
            return self.random_policy.predict_actions(df, budget=budget)
        if policy_name == "simple_recency_rule":
            return self.recency_policy.predict_actions(df, budget=budget)
        if policy_name == "recency_category_affinity":
            return self.recency_policy.predict_actions(
                df, budget=budget, mode_override="category_affinity"
            )
        if policy_name == "regularized_logistic":
            return self.regularized_policy.predict_actions(df, budget=budget)
        if policy_name == "flexible_hist_gbdt":
            return self.flexible_policy.predict_actions(df, budget=budget)
        raise ValueError(f"Unknown policy_name: {policy_name}")


def fit_and_freeze_policies(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    budget: float = 0.25,
    seed: int = 42,
) -> FrozenPolicySuite:
    """Fit action-response models on train_df, select hyperparameters on val_df, and freeze."""
    validate_feature_columns(PRE_CAMPAIGN_FEATURES)
    val_no_email = np.full(len(val_df), "No E-Mail", dtype=object)

    # 1. Simple Recency Rule: select between category_affinity and mens_default on val_df
    recency_history: list[dict[str, Any]] = []
    best_recency_mode = "category_affinity"
    best_recency_val = -np.inf
    temp_recency = SimpleRecencyPolicy(email_mode="category_affinity", val_tuning_history=[])
    for mode in ("category_affinity", "mens_default"):
        val_acts = temp_recency.predict_actions(val_df, budget=budget, mode_override=mode)
        comp = compare_policies_paired(val_df, val_acts, val_no_email, outcome="visit")
        recency_history.append(
            {
                "email_mode": mode,
                "val_incremental_visits_per_1k": round(comp["diff"], 2),
                "val_ci95_low": round(comp["ci95_low"], 2),
                "val_ci95_high": round(comp["ci95_high"], 2),
            }
        )
        if comp["diff"] > best_recency_val:
            best_recency_val = comp["diff"]
            best_recency_mode = mode

    frozen_recency = SimpleRecencyPolicy(
        email_mode=best_recency_mode, val_tuning_history=recency_history
    )

    # 2. Regularized Model: L2-regularized LogisticRegression per action
    pre_lr = ColumnTransformer(
        [
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), list(CATEGORICAL_FEATURES)),
        ]
    ).fit(train_df[list(PRE_CAMPAIGN_FEATURES)])
    X_tr_lr = pre_lr.transform(train_df[list(PRE_CAMPAIGN_FEATURES)])

    lr_history: list[dict[str, Any]] = []
    best_c = 0.01
    best_lr_val = -np.inf
    best_lr_models: dict[str, LogisticRegression] = {}

    for c_val in (0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0):
        candidate_models: dict[str, LogisticRegression] = {}
        for act in VALID_ACTIONS:
            mask = (train_df["segment"] == act).to_numpy()
            clf = LogisticRegression(C=c_val, max_iter=1000, random_state=seed)
            clf.fit(X_tr_lr[mask], train_df.loc[mask, "visit"].to_numpy())
            candidate_models[act] = clf
        cand_policy = RegularizedActionResponsePolicy(
            preprocessor=pre_lr,
            models=candidate_models,
            best_c=c_val,
            val_tuning_history=[],
        )
        val_acts = cand_policy.predict_actions(val_df, budget=budget)
        comp = compare_policies_paired(val_df, val_acts, val_no_email, outcome="visit")
        lr_history.append(
            {
                "C": c_val,
                "val_incremental_visits_per_1k": round(comp["diff"], 2),
                "val_ci95_low": round(comp["ci95_low"], 2),
                "val_ci95_high": round(comp["ci95_high"], 2),
            }
        )
        if comp["diff"] > best_lr_val:
            best_lr_val = comp["diff"]
            best_c = c_val
            best_lr_models = candidate_models

    frozen_lr = RegularizedActionResponsePolicy(
        preprocessor=pre_lr,
        models=best_lr_models,
        best_c=best_c,
        val_tuning_history=lr_history,
    )

    # 3. Flexible Model: HistGradientBoostingClassifier per action
    pre_gb = ColumnTransformer(
        [
            ("num", "passthrough", list(NUMERIC_FEATURES)),
            ("cat", OneHotEncoder(sparse_output=False), list(CATEGORICAL_FEATURES)),
        ]
    ).fit(train_df[list(PRE_CAMPAIGN_FEATURES)])
    X_tr_gb = pre_gb.transform(train_df[list(PRE_CAMPAIGN_FEATURES)])

    gb_history: list[dict[str, Any]] = []
    best_gb_params: dict[str, Any] = {}
    best_gb_val = -np.inf
    best_gb_models: dict[str, HistGradientBoostingClassifier] = {}

    for max_depth in (2, 3, 6):
        for min_leaf in (50, 200):
            params = {
                "max_depth": max_depth,
                "min_samples_leaf": min_leaf,
                "l2_regularization": 1.0,
                "learning_rate": 0.05,
                "max_iter": 100,
            }
            candidate_gb: dict[str, HistGradientBoostingClassifier] = {}
            for act in VALID_ACTIONS:
                mask = (train_df["segment"] == act).to_numpy()
                gb = HistGradientBoostingClassifier(**params, random_state=seed)
                gb.fit(X_tr_gb[mask], train_df.loc[mask, "visit"].to_numpy())
                candidate_gb[act] = gb
            cand_gb_policy = FlexibleActionResponsePolicy(
                preprocessor=pre_gb,
                models=candidate_gb,
                best_params=params,
                val_tuning_history=[],
            )
            val_acts = cand_gb_policy.predict_actions(val_df, budget=budget)
            comp = compare_policies_paired(val_df, val_acts, val_no_email, outcome="visit")
            gb_history.append(
                {
                    **params,
                    "val_incremental_visits_per_1k": round(comp["diff"], 2),
                    "val_ci95_low": round(comp["ci95_low"], 2),
                    "val_ci95_high": round(comp["ci95_high"], 2),
                }
            )
            if comp["diff"] > best_gb_val:
                best_gb_val = comp["diff"]
                best_gb_params = params
                best_gb_models = candidate_gb

    frozen_gb = FlexibleActionResponsePolicy(
        preprocessor=pre_gb,
        models=best_gb_models,
        best_params=best_gb_params,
        val_tuning_history=gb_history,
    )

    return FrozenPolicySuite(
        random_policy=RandomTargetingPolicy(seed=seed),
        recency_policy=frozen_recency,
        regularized_policy=frozen_lr,
        flexible_policy=frozen_gb,
    )


def summarize_targeting_composition(
    df: pd.DataFrame, actions: np.ndarray
) -> dict[str, Any]:
    n = len(actions)
    contacted_mask = actions != "No E-Mail"
    mens_mask = actions == "Mens E-Mail"
    womens_mask = actions == "Womens E-Mail"
    contacted_df = df.loc[contacted_mask]
    return {
        "n_eligible": int(n),
        "n_contacted": int(np.sum(contacted_mask)),
        "contact_rate": round(float(np.mean(contacted_mask)), 4),
        "mens_email_share_of_eligible": round(float(np.mean(mens_mask)), 4),
        "womens_email_share_of_eligible": round(float(np.mean(womens_mask)), 4),
        "no_email_share_of_eligible": round(float(np.mean(actions == "No E-Mail")), 4),
        "contacted_mean_recency": round(float(contacted_df["recency"].mean()), 2)
        if len(contacted_df) > 0
        else None,
        "contacted_mean_history_spend": round(float(contacted_df["history"].mean()), 2)
        if len(contacted_df) > 0
        else None,
        "contacted_both_categories_share": round(
            float(((contacted_df["mens"] == 1) & (contacted_df["womens"] == 1)).mean()), 4
        )
        if len(contacted_df) > 0
        else None,
    }


def run_full_study(
    csv_path: Path | str,
    sql_path: Path | str,
    seed: int = 42,
) -> dict[str, Any]:
    df = load_dataset(csv_path, verify_checksum=True)
    audit = run_duckdb_audit(csv_path, sql_path)
    train_df, val_df, test_df = make_splits(df, seed=seed)
    frozen = fit_and_freeze_policies(train_df, val_df, budget=0.25, seed=seed)

    policy_names = (
        "no_email",
        "random_targeting",
        "simple_recency_rule",
        "recency_category_affinity",
        "regularized_logistic",
        "flexible_hist_gbdt",
    )
    policy_labels = {
        "no_email": "No E-Mail (0% Contact Baseline)",
        "random_targeting": "Random Targeting (25% Budget)",
        "simple_recency_rule": f"Simple Recency Rule ({frozen.recency_policy.email_mode})",
        "recency_category_affinity": "Simple Recency + Category Match Rule",
        "regularized_logistic": f"Regularized Logistic T-Learner (C={frozen.regularized_policy.best_c})",
        "flexible_hist_gbdt": (
            f"Flexible HistGBDT T-Learner "
            f"(depth={frozen.flexible_policy.best_params['max_depth']}, "
            f"leaf={frozen.flexible_policy.best_params['min_samples_leaf']})"
        ),
    }

    # Evaluate primary 25% contact budget across visit, conversion, and spend on held-out test_df
    actions_25 = {
        name: frozen.predict_actions(name, test_df, budget=0.25) for name in policy_names
    }
    no_email_25 = actions_25["no_email"]
    random_25 = actions_25["random_targeting"]
    best_simple_25 = actions_25["simple_recency_rule"]

    primary_25_results: list[dict[str, Any]] = []
    for name in policy_names:
        acts = actions_25[name]
        visit_abs = evaluate_policy_ipw(test_df, acts, outcome="visit", scale=1000.0)
        visit_vs_none = compare_policies_paired(
            test_df, acts, no_email_25, outcome="visit", scale=1000.0
        )
        visit_vs_random = compare_policies_paired(
            test_df, acts, random_25, outcome="visit", scale=1000.0
        )
        visit_vs_simple = compare_policies_paired(
            test_df, acts, best_simple_25, outcome="visit", scale=1000.0
        )

        conv_abs = evaluate_policy_ipw(test_df, acts, outcome="conversion", scale=1000.0)
        conv_vs_none = compare_policies_paired(
            test_df, acts, no_email_25, outcome="conversion", scale=1000.0
        )
        conv_vs_simple = compare_policies_paired(
            test_df, acts, best_simple_25, outcome="conversion", scale=1000.0
        )

        spend_abs = evaluate_policy_ipw(test_df, acts, outcome="spend", scale=1000.0)
        spend_vs_none = compare_policies_paired(
            test_df, acts, no_email_25, outcome="spend", scale=1000.0
        )
        spend_vs_simple = compare_policies_paired(
            test_df, acts, best_simple_25, outcome="spend", scale=1000.0
        )

        primary_25_results.append(
            {
                "policy": name,
                "label": policy_labels[name],
                "composition": summarize_targeting_composition(test_df, acts),
                "visits_per_1k": visit_abs,
                "incremental_visits_vs_no_email": visit_vs_none,
                "incremental_visits_vs_random": visit_vs_random,
                "incremental_visits_vs_best_simple": visit_vs_simple,
                "conversions_per_1k": conv_abs,
                "incremental_conversions_vs_no_email": conv_vs_none,
                "incremental_conversions_vs_best_simple": conv_vs_simple,
                "spend_per_1k": spend_abs,
                "incremental_spend_vs_no_email": spend_vs_none,
                "incremental_spend_vs_best_simple": spend_vs_simple,
            }
        )

    # Sensitivity analysis across 10%, 25%, and 50% contact budgets
    budget_sensitivity: list[dict[str, Any]] = []
    for b in (0.10, 0.25, 0.50):
        acts_b = {
            name: frozen.predict_actions(name, test_df, budget=b)
            for name in (
                "random_targeting",
                "simple_recency_rule",
                "recency_category_affinity",
                "regularized_logistic",
                "flexible_hist_gbdt",
            )
        }
        best_simple_b = acts_b["simple_recency_rule"]
        for name, acts in acts_b.items():
            v_abs = evaluate_policy_ipw(test_df, acts, outcome="visit", scale=1000.0)
            v_none = compare_policies_paired(
                test_df, acts, no_email_25, outcome="visit", scale=1000.0
            )
            v_simp = compare_policies_paired(
                test_df, acts, best_simple_b, outcome="visit", scale=1000.0
            )
            budget_sensitivity.append(
                {
                    "budget": b,
                    "budget_pct": int(round(b * 100)),
                    "policy": name,
                    "label": policy_labels[name],
                    "visits_per_1k": round(v_abs["estimate"], 2),
                    "inc_visits_vs_no_email": round(v_none["diff"], 2),
                    "inc_visits_vs_no_email_ci95": [
                        round(v_none["ci95_low"], 2),
                        round(v_none["ci95_high"], 2),
                    ],
                    "inc_visits_vs_best_simple": round(v_simp["diff"], 2),
                    "inc_visits_vs_best_simple_ci95": [
                        round(v_simp["ci95_low"], 2),
                        round(v_simp["ci95_high"], 2),
                    ],
                    "disagreement_with_best_simple": round(v_simp["disagreement_rate"], 4),
                }
            )

    return {
        "dataset": {
            "path": "data/hillstrom-email-experiment.csv",
            "sha256": EXPECTED_SHA256,
            "total_rows": len(df),
            "splits": {
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "seed": seed,
            },
            "documented_assignment_probabilities": ASSIGNMENT_PROBABILITIES,
            "pre_campaign_features": list(PRE_CAMPAIGN_FEATURES),
            "excluded_columns": list(FORBIDDEN_FEATURE_COLUMNS),
        },
        "duckdb_audit": audit,
        "validation_tuning": {
            "simple_recency_rule": {
                "selected_email_mode": frozen.recency_policy.email_mode,
                "candidates": frozen.recency_policy.val_tuning_history,
            },
            "regularized_logistic": {
                "selected_C": frozen.regularized_policy.best_c,
                "candidates": frozen.regularized_policy.val_tuning_history,
            },
            "flexible_hist_gbdt": {
                "selected_params": frozen.flexible_policy.best_params,
                "candidates": frozen.flexible_policy.val_tuning_history,
            },
        },
        "held_out_test_25pct_budget": primary_25_results,
        "budget_sensitivity": budget_sensitivity,
    }


def render_policy_value_svg(results: dict[str, Any], out_path: Path | str) -> None:
    """Render a deterministic two-panel SVG chart of held-out policy values with 95% CIs."""
    width, height = 980, 500
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="system-ui, -apple-system, sans-serif">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="490" y="30" text-anchor="middle" font-size="17" font-weight="700" fill="#111827">Held-Out Policy Evaluation on Hillstrom Experiment (N_test = 16,000, Documented IPW p = 1/3)</text>',
        '<text x="490" y="50" text-anchor="middle" font-size="12" fill="#4b5563">Incremental 2-Week Site Visits per 1,000 Eligible Customers vs. No E-Mail (Paired 95% Confidence Intervals)</text>',
    ]

    # Panel A (Left): 25% Budget Policy Comparison (Horizontal forest plot)
    p1_x0, p1_x1 = 215, 465
    p1_y0, p1_y1 = 95, 420
    lines.append('<text x="245" y="78" text-anchor="middle" font-size="13" font-weight="600" fill="#1f2937">A. 25% Contact Budget — Policy Comparison</text>')
    lines.append(f'<rect x="{p1_x0}" y="{p1_y0}" width="{p1_x1 - p1_x0}" height="{p1_y1 - p1_y0}" fill="#f9fafb" stroke="#d1d5db"/>')

    # X scale for Panel A: -5 to +35 visits / 1,000
    def x_scale_a(val: float) -> float:
        return p1_x0 + ((val - (-5.0)) / 40.0) * (p1_x1 - p1_x0)

    for tick in (-5, 0, 10, 20, 30):
        tx = x_scale_a(float(tick))
        stroke = "#ef4444" if tick == 0 else "#e5e7eb"
        dash = ' stroke-dasharray="4,3"' if tick == 0 else ""
        lines.append(f'<line x1="{tx:.1f}" y1="{p1_y0}" x2="{tx:.1f}" y2="{p1_y1}" stroke="{stroke}" stroke-width="1"{dash}/>')
        lines.append(f'<text x="{tx:.1f}" y="{p1_y1 + 18}" text-anchor="middle" font-size="11" fill="#374151">{tick:+d}</text>')
    lines.append(f'<text x="{(p1_x0 + p1_x1) / 2:.1f}" y="{p1_y1 + 38}" text-anchor="middle" font-size="11" font-weight="600" fill="#374151">Incremental Visits / 1,000 vs. No E-Mail (95% CI)</text>')

    panel_a_items = [
        r for r in results["held_out_test_25pct_budget"] if r["policy"] != "no_email"
    ]
    colors = {
        "random_targeting": "#6b7280",
        "recency_category_affinity": "#0284c7",
        "simple_recency_rule": "#2563eb",
        "regularized_logistic": "#059669",
        "flexible_hist_gbdt": "#7c3aed",
    }
    short_names = {
        "random_targeting": "Random Targeting (25%)",
        "recency_category_affinity": "Recency + Category Match",
        "simple_recency_rule": "Simple Recency (Val-Tuned)",
        "regularized_logistic": "Regularized Logistic (C=0.01)",
        "flexible_hist_gbdt": "Flexible HistGBDT (d=3)",
    }

    for idx, item in enumerate(panel_a_items):
        cy = p1_y0 + 38 + idx * 62
        inc = item["incremental_visits_vs_no_email"]
        est = inc["diff"]
        lo = inc["ci95_low"]
        hi = inc["ci95_high"]
        col = colors.get(item["policy"], "#111827")
        label = short_names.get(item["policy"], item["policy"])
        x_est = x_scale_a(est)
        x_lo = x_scale_a(lo)
        x_hi = x_scale_a(hi)

        lines.append(f'<text x="{p1_x0 - 10}" y="{cy + 4}" text-anchor="end" font-size="11" font-weight="600" fill="#1f2937">{label}</text>')
        lines.append(f'<line x1="{x_lo:.1f}" y1="{cy}" x2="{x_hi:.1f}" y2="{cy}" stroke="{col}" stroke-width="2.5"/>')
        lines.append(f'<line x1="{x_lo:.1f}" y1="{cy - 5}" x2="{x_lo:.1f}" y2="{cy + 5}" stroke="{col}" stroke-width="2"/>')
        lines.append(f'<line x1="{x_hi:.1f}" y1="{cy - 5}" x2="{x_hi:.1f}" y2="{cy + 5}" stroke="{col}" stroke-width="2"/>')
        lines.append(f'<circle cx="{x_est:.1f}" cy="{cy}" r="5" fill="{col}"/>')
        lines.append(
            f'<text x="{x_est:.1f}" y="{cy - 10}" text-anchor="middle" font-size="10" font-weight="600" fill="{col}">{est:+.1f} [{lo:+.1f}, {hi:+.1f}]</text>'
        )

    # Panel B (Right): Budget Sensitivity (10%, 25%, 50%)
    p2_x0, p2_x1 = 555, 940
    p2_y0, p2_y1 = 95, 420
    lines.append('<text x="747" y="78" text-anchor="middle" font-size="13" font-weight="600" fill="#1f2937">B. Contact Budget Sensitivity (10%, 25%, 50%)</text>')
    lines.append(f'<rect x="{p2_x0}" y="{p2_y0}" width="{p2_x1 - p2_x0}" height="{p2_y1 - p2_y0}" fill="#f9fafb" stroke="#d1d5db"/>')

    def y_scale_b(val: float) -> float:
        return p2_y1 - ((val - 0.0) / 62.0) * (p2_y1 - p2_y0)

    for y_tick in (0, 10, 20, 30, 40, 50, 60):
        ty = y_scale_b(float(y_tick))
        lines.append(f'<line x1="{p2_x0}" y1="{ty:.1f}" x2="{p2_x1}" y2="{ty:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{p2_x0 - 8}" y="{ty + 4:.1f}" text-anchor="end" font-size="11" fill="#374151">+{y_tick}</text>')

    budget_x = {10: p2_x0 + 65, 25: (p2_x0 + p2_x1) / 2, 50: p2_x1 - 65}
    for b_pct, bx in budget_x.items():
        lines.append(f'<line x1="{bx:.1f}" y1="{p2_y0}" x2="{bx:.1f}" y2="{p2_y1}" stroke="#f3f4f6" stroke-width="1"/>')
        lines.append(f'<text x="{bx:.1f}" y="{p2_y1 + 18}" text-anchor="middle" font-size="11" font-weight="600" fill="#374151">{b_pct}% Budget</text>')
    lines.append(f'<text x="{(p2_x0 + p2_x1) / 2:.1f}" y="{p2_y1 + 38}" text-anchor="middle" font-size="11" font-weight="600" fill="#374151">Contact Budget (% of Eligible Customers)</text>')

    plotted_policies = [
        ("random_targeting", -9.0),
        ("simple_recency_rule", -3.0),
        ("regularized_logistic", +3.0),
        ("flexible_hist_gbdt", +9.0),
    ]
    sens_rows = results["budget_sensitivity"]
    for pol_name, x_offset in plotted_policies:
        col = colors[pol_name]
        pol_pts = [r for r in sens_rows if r["policy"] == pol_name]
        pol_pts.sort(key=lambda r: r["budget_pct"])
        coords: list[tuple[float, float]] = []
        for pt in pol_pts:
            bx = budget_x[pt["budget_pct"]] + x_offset
            by = y_scale_b(pt["inc_visits_vs_no_email"])
            ylo = y_scale_b(pt["inc_visits_vs_no_email_ci95"][0])
            yhi = y_scale_b(pt["inc_visits_vs_no_email_ci95"][1])
            coords.append((bx, by))
            lines.append(f'<line x1="{bx:.1f}" y1="{ylo:.1f}" x2="{bx:.1f}" y2="{yhi:.1f}" stroke="{col}" stroke-width="2"/>')
            lines.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="4.5" fill="{col}"/>')
        if len(coords) >= 2:
            poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
            lines.append(f'<polyline fill="none" stroke="{col}" stroke-width="2" points="{poly}"/>')

    # Legend inside Panel B top-left
    leg_x, leg_y = p2_x0 + 14, p2_y0 + 18
    lines.append(f'<rect x="{leg_x - 6}" y="{leg_y - 12}" width="215" height="78" fill="#ffffff" stroke="#e5e7eb" rx="4"/>')
    for idx, (pol_name, _) in enumerate(plotted_policies):
        ly = leg_y + idx * 18
        col = colors[pol_name]
        lbl = short_names[pol_name]
        lines.append(f'<line x1="{leg_x}" y1="{ly - 3}" x2="{leg_x + 16}" y2="{ly - 3}" stroke="{col}" stroke-width="2.5"/>')
        lines.append(f'<circle cx="{leg_x + 8}" cy="{ly - 3}" r="3.5" fill="{col}"/>')
        lines.append(f'<text x="{leg_x + 24}" y="{ly}" font-size="10.5" fill="#1f2937">{lbl}</text>')

    lines.append("</svg>")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
