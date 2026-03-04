"""
bandit_data_pipeline.py

Creates offline contextual bandit datasets for three UCI classification tasks:
  • Drug Consumption   (4-arm bandit, sensitive: gender, education)
  • Student Performance (4-arm bandit, sensitive: parental education)
  • Adult Income        (2-arm bandit, sensitive: gender, race)

Offline data format per sample: (x, a, r(x,a), pi_b(a|x))
  x           : context feature vector
  a           : action chosen by the logging policy (class index 0..K-1)
  r(x,a)      : reward  = +1 if action matches ground-truth label, -1 otherwise
  pi_b(a|x)   : propensity score — probability of choosing a under the logging policy

Logging policies
  - Random  : uniform over all K actions  (pi_b = 1/K)
  - Tweak-1 : one fixed action with prob rho=0.9, remaining (1-rho) split evenly
  - Mixed   : logistic regression trained on the training split (realistic but suboptimal)

Train / test split: 70% / 30%  (stratified by true label)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import os, ipdb, sys, math
import joblib

from data_utils import load_drugs_data, load_student_data, load_adult_data, load_loan_data, load_dataset

# ─── Logging Policies ────────────────────────────────────────────────────────

def random_policy(n_samples: int, n_actions: int):
    """Uniform random: pi_b(a|x) = 1/K for all a."""
    actions      = np.random.randint(0, n_actions, size=n_samples)
    propensities = np.full(n_samples, 1.0 / n_actions)
    return actions, propensities


def tweak1_policy(n_samples: int, n_actions: int,
                  fixed_action: int = 0, rho: float = 0.9):
    """
    Biased policy: `fixed_action` is chosen with probability rho;
    remaining (1-rho) is split equally among all other actions.
    """
    other_prob = (1.0 - rho) / (n_actions - 1)
    probs       = np.full(n_actions, other_prob)
    probs[fixed_action] = rho

    actions      = np.random.choice(n_actions, size=n_samples, p=probs)
    propensities = np.where(actions == fixed_action, rho, other_prob)
    return actions, propensities


def train_mixed_policy(X_train: np.ndarray, y_train: np.ndarray):
    """
    Fit a regularised logistic regression as a realistic-but-suboptimal
    logging policy. Returns (fitted_clf, fitted_scaler).
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=500, C=0.5, random_state=42)
    clf.fit(X_scaled, y_train)
    return clf, scaler


def apply_mixed_policy(clf, scaler, X: np.ndarray, n_actions: int):
    """
    Sample one action per context from the learned policy's probability
    distribution. Columns of predict_proba are aligned to class indices
    0..K-1 regardless of the internal ordering in clf.classes_.
    """
    X_scaled = scaler.transform(X)
    probs    = clf.predict_proba(X_scaled)          # (n, len(clf.classes_))

    full_probs = np.zeros((len(X), n_actions))
    for col_idx, class_idx in enumerate(clf.classes_):
        full_probs[:, class_idx] = probs[:, col_idx]

    # Numerical safety: clip and renormalise
    full_probs  = np.clip(full_probs, 1e-8, None)
    full_probs /= full_probs.sum(axis=1, keepdims=True)

    actions      = np.array([np.random.choice(n_actions, p=full_probs[i])
                              for i in range(len(X))])
    propensities = full_probs[np.arange(len(actions)), actions]
    return actions, propensities


# ─── Dataset Assembler ───────────────────────────────────────────────────────

def make_bandit_df(X: np.ndarray, y_true: np.ndarray, sensitive: pd.DataFrame,
                   actions: np.ndarray, propensities: np.ndarray,
                   feature_cols: list) -> dict:
    """
    Assemble offline bandit data as a structured dict.

    Returns
    -------
    {
      'context'  : DataFrame (n, d)  — feature vectors
      'actions'  : DataFrame         — columns: action, propensity, true_label
      'reward'   : Series            — +1 / -1
      'sensitive': DataFrame         — sensitive attribute column(s)
    }
    """
    rewards = np.where(actions == y_true, 1, -1)
    return {
        'context':   pd.DataFrame(X, columns=feature_cols),
        'actions':   pd.DataFrame({'action':     actions,
                                   'propensity': propensities,
                                   'true_label': y_true}),
        'reward':    pd.Series(rewards, name='reward'),
        'sensitive': sensitive.reset_index(drop=True),
    }


# ─── Domain-specific Mixed Policies ─────────────────────────────────────────

def loan_historical_policy(X: np.ndarray, feature_cols: list,
                           epsilon: float = 0.1):
    """
    Domain-informed logging policy for German Credit (loan officer heuristic).

    Predicts 'good' (action=1) when the applicant's credit history shows prior
    repayment (A30/A31/A32), and 'bad' (action=0) for delay or critical
    accounts (A33/A34).  ε-greedy smoothing (ε=0.1) ensures non-zero
    propensities required for IPS estimation.

    Corresponds to CreditBanditFactory.historical_policy but adapted to our
    one-hot encoded feature matrix and 0/1 action space.

    propensity of chosen action:
        (1 - ε) + ε/K  if action matches the deterministic decision
        ε/K            otherwise
    """
    n_actions = 2
    idx = {col: i for i, col in enumerate(feature_cols)}
    good_history = (
        X[:, idx['credit_history_A30']] +
        X[:, idx['credit_history_A31']] +
        X[:, idx['credit_history_A32']]
    ) > 0
    det_actions = (~good_history).astype(int)  # 0 = good (lend), 1 = bad (reject)

    # ε-greedy: explore uniformly with probability ε
    explore  = np.random.rand(len(X)) < epsilon
    actions  = np.where(explore, np.random.randint(0, n_actions, len(X)), det_actions)

    main_prob  = (1.0 - epsilon) + epsilon / n_actions   # prob of deterministic action
    other_prob = epsilon / n_actions
    propensities = np.where(actions == det_actions, main_prob, other_prob)

    return actions, propensities


# ─── Generic Pipeline ────────────────────────────────────────────────────────

def _build_bandit_datasets(X: np.ndarray, y: np.ndarray,
                           sensitive: pd.DataFrame, feature_cols: list,
                           class_names: list, dataset_name: str,
                           sensitive_attr: str,
                           train_frac: float = 0.7,
                           rho: float = 0.9, seed: int = 42,
                           mixed_policy_fn=None,
                           save_dir: str = None) -> dict:
    """
    Core pipeline shared by all datasets.

    Parameters
    ----------
    X            : (n, d) float32 context features
    y            : (n,)   integer class indices
    sensitive    : (n, *)  DataFrame of ALL sensitive attribute columns
    feature_cols : list of feature column names (length d)
    class_names  : ordered list of class label strings (length K)
    dataset_name : display name used in printed summaries
    sensitive_attr   : base name of the sensitive attribute to use (e.g. 'gender').
                       Must satisfy f'{sensitive_attr}_label' ∈ sensitive.columns.
                       All other sensitive columns are folded back into X as features.
    train_frac       : proportion of data used for training (0 < train_frac < 1).
                       The remainder is split evenly between safety and test.
    rho              : Tweak-1 bias probability
    seed             : global random seed
    mixed_policy_fn  : optional callable(X, feature_cols) -> (actions, propensities)
                       If provided, replaces the default logistic-regression mixed
                       policy.  Used for domain-specific policies (e.g. loan officer).
    save_dir         : root directory for saving outputs. CSVs are written to
                       {save_dir}/{dataset_slug}/{sensitive_attr}/.
                       The fitted mixed policy (if logistic regression) is saved as
                       mixed_policy.joblib in the same directory.
                       Pass None to disable saving.

    Returns
    -------
    dict  {policy_name: {'train': pd.DataFrame, 'safety': pd.DataFrame, 'test': pd.DataFrame}}
          policy_name in {'random', 'tweak1', 'mixed'}
    """
    # ── Validate and select active sensitive attribute ─────────────────────
    label_col = f'{sensitive_attr}_label'
    if label_col not in sensitive.columns:
        valid = [c.removesuffix('_label') for c in sensitive.columns]
        raise ValueError(
            f"'{label_col}' not found in sensitive attributes {list(sensitive.columns)}. "
            f"Valid choices: {valid}"
        )

    other_sens_cols = [c for c in sensitive.columns if c != label_col]
    active_sensitive = sensitive[[label_col]].reset_index(drop=True)

    # Fold other sensitive columns back into X so they remain as context features
    if other_sens_cols:
        other_vals = sensitive[other_sens_cols].values.astype(np.float32)
        X          = np.concatenate([X, other_vals], axis=1)
        feature_cols = feature_cols + other_sens_cols

    # ── Output directory ──────────────────────────────────────────────────
    if save_dir is not None:
        dataset_slug = (dataset_name.lower()
                        .replace(' ', '_').replace('(', '').replace(')', ''))
        out_dir = os.path.join(save_dir, dataset_slug, sensitive_attr)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = None

    np.random.seed(seed)
    n_actions = len(class_names)

    print(f"\n{'='*65}")
    print(f"  {dataset_name}  ({len(X)} samples | {n_actions} actions)")
    print(f"{'='*65}")
    print(f"  Classes : {class_names}")

    # ── Stratified three-way split: train / safety / test ────────────────
    holdout_frac = 1.0 - train_frac          # safety + test combined
    half_holdout = holdout_frac / 2.0        # each of safety and test

    idx = np.arange(len(X))
    train_idx, holdout_idx = train_test_split(
        idx, test_size=holdout_frac, random_state=seed, stratify=y
    )
    try:
        safety_idx, test_idx = train_test_split(
            holdout_idx, test_size=0.5, random_state=seed, stratify=y[holdout_idx]
        )
    except ValueError:
        safety_idx, test_idx = train_test_split(
            holdout_idx, test_size=0.5, random_state=seed
        )
    print(f"  train_frac = {train_frac:.2f}  →  "
          f"train: {len(train_idx)} | safety: {len(safety_idx)} | test: {len(test_idx)}"
          f"  (safety/test each ≈ {half_holdout:.2f})")

    # ── Fit mixed logging policy on training split ─────────────────────────
    if mixed_policy_fn is None:
        clf, scaler = train_mixed_policy(X[train_idx], y[train_idx])
        if out_dir is not None:
            joblib.dump({'clf': clf, 'scaler': scaler},
                        os.path.join(out_dir, 'mixed_policy.joblib'))
            print(f"  Saved mixed policy → {os.path.join(out_dir, 'mixed_policy.joblib')}")

    # ── Generate offline datasets ─────────────────────────────────────────
    datasets: dict = {}

    for policy_name in ('random', 'tweak1', 'mixed'):
        split_dfs: dict = {}

        splits = (('train', train_idx), ('safety', safety_idx), ('test', test_idx))
        for split_name, split_idx in splits:
            X_s    = X[split_idx]
            y_s    = y[split_idx]
            sens_s = active_sensitive.iloc[split_idx].reset_index(drop=True)

            if policy_name == 'random':
                actions, props = random_policy(len(split_idx), n_actions)
            elif policy_name == 'tweak1':
                actions, props = tweak1_policy(len(split_idx), n_actions, rho=rho)
            elif mixed_policy_fn is not None:
                actions, props = mixed_policy_fn(X_s, feature_cols)
            else:
                actions, props = apply_mixed_policy(clf, scaler, X_s, n_actions)

            split_dfs[split_name] = make_bandit_df(
                X_s, y_s, sens_s, actions, props, feature_cols
            )

        datasets[policy_name] = split_dfs

        # ── Summary stats ─────────────────────────────────────────────────
        tr, sf, te = split_dfs['train'], split_dfs['safety'], split_dfs['test']
        print(f"\n  [{policy_name.upper()} Policy]")
        for label, d in (('Train ', tr), ('Safety', sf), ('Test  ', te)):
            print(f"    {label}: {len(d['reward']):>6} samples | "
                  f"reward mean = {d['reward'].mean():.3f} | "
                  f"mean propensity = {d['actions']['propensity'].mean():.3f}")

        # Per-group reward means for the active sensitive attribute
        for col in active_sensitive.columns:
            for grp_val in sorted(active_sensitive[col].unique()):
                mask = tr['sensitive'][col] == grp_val
                n    = mask.sum()
                mean = tr['reward'][mask].mean()
                print(f"      {col}={grp_val:<18} (n={n:>5}) "
                      f"reward mean = {mean:.3f}")

    # ── Save full nested dict as a single file ────────────────────────────
    if out_dir is not None:
        path = os.path.join(out_dir, 'datasets.joblib')
        joblib.dump(datasets, path)
        print(f"\n  Saved datasets → {path}")

    return datasets


# ─── Dataset-Specific Wrappers ───────────────────────────────────────────────

def build_drug_bandit_datasets(sensitive_attr: str, train_frac: float = 0.7,
                               rho: float = 0.9, seed: int = 42,
                               save_dir: str = None) -> dict:
    """
    Offline bandit datasets for the Drug Consumption task.

    4-arm bandit: {'10 years', '5 years', 'past year', 'past month'}
    Sensitive attributes: gender (M/F), education (above/below college)
    """
    df, meta = load_drugs_data()
    df = df.reset_index(drop=True)

    non_feature  = {meta['target_col']} | set(meta['sensitive_cols'])
    feature_cols = [c for c in df.columns if c not in non_feature]

    X         = df[feature_cols].values.astype(np.float32)
    y         = df[meta['target_col']].values
    sensitive = df[meta['sensitive_cols']].reset_index(drop=True)

    return _build_bandit_datasets(
        X, y, sensitive, feature_cols, meta['class_names'],
        'Drug Consumption', sensitive_attr=sensitive_attr,
        train_frac=train_frac, rho=rho, seed=seed, save_dir=save_dir
    )


def build_student_bandit_datasets(sensitive_attr: str, train_frac: float = 0.7,
                                  rho: float = 0.9, seed: int = 42,
                                  save_dir: str = None) -> dict:
    """
    Offline bandit datasets for the Student Performance task.

    5-arm bandit: {'fail', 'sufficient', 'satisfactory', 'good', 'excellent'}
    Sensitive attribute: parental_edu_label (above/below college)
    """
    df, meta = load_student_data(subject='por')
    df = df.reset_index(drop=True)

    non_feature  = {meta['target_col']} | set(meta['sensitive_cols'])
    feature_cols = [c for c in df.columns if c not in non_feature]

    X         = df[feature_cols].values.astype(np.float32)
    y         = df[meta['target_col']].values
    sensitive = df[meta['sensitive_cols']].reset_index(drop=True)

    return _build_bandit_datasets(
        X, y, sensitive, feature_cols, meta['class_names'],
        'Student Performance', sensitive_attr=sensitive_attr,
        train_frac=train_frac, rho=rho, seed=seed, save_dir=save_dir
    )


def build_adult_bandit_datasets(sensitive_attr: str, train_frac: float = 0.7,
                                rho: float = 0.9, seed: int = 42,
                                save_dir: str = None) -> dict:
    """
    Offline bandit datasets for the Adult Income task.

    2-arm bandit: {'<=50K', '>50K'}
    Sensitive attributes: gender_label (Male/Female), race_label (White/Non-White)
    """
    df, meta = load_adult_data()
    df = df.reset_index(drop=True)

    non_feature  = {meta['target_col']} | set(meta['sensitive_cols'])
    feature_cols = [c for c in df.columns if c not in non_feature]

    X         = df[feature_cols].values.astype(np.float32)
    y         = df[meta['target_col']].values
    sensitive = df[meta['sensitive_cols']].reset_index(drop=True)

    return _build_bandit_datasets(
        X, y, sensitive, feature_cols, meta['class_names'],
        'Adult Income', sensitive_attr=sensitive_attr,
        train_frac=train_frac, rho=rho, seed=seed, save_dir=save_dir
    )


def build_loan_bandit_datasets(sensitive_attr: str, train_frac: float = 0.7,
                               rho: float = 0.9, seed: int = 42,
                               save_dir: str = None) -> dict:
    """
    Offline bandit datasets for the Statlog German Credit task.

    2-arm bandit: {'good', 'bad'}  (credit risk; label 0=good, 1=bad)
    Sensitive attributes: sex_label (Male/Female), age_label (young/old, threshold 25)
    """
    df, meta = load_loan_data()
    df = df.reset_index(drop=True)

    non_feature  = {meta['target_col']} | set(meta['sensitive_cols'])
    feature_cols = [c for c in df.columns if c not in non_feature]

    X         = df[feature_cols].values.astype(np.float32)
    y         = df[meta['target_col']].values
    sensitive = df[meta['sensitive_cols']].reset_index(drop=True)

    return _build_bandit_datasets(
        X, y, sensitive, feature_cols, meta['class_names'],
        'German Credit (Loan)', sensitive_attr=sensitive_attr,
        train_frac=train_frac, rho=rho, seed=seed,
        mixed_policy_fn=loan_historical_policy, save_dir=save_dir
    )


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAVE_DIR = None
    drug_data    = build_drug_bandit_datasets(sensitive_attr='gender', save_dir=SAVE_DIR)
    student_data = build_student_bandit_datasets(sensitive_attr='parental_edu', save_dir=SAVE_DIR)
    adult_data   = build_adult_bandit_datasets(sensitive_attr='gender', save_dir=SAVE_DIR)
    loan_data    = build_loan_bandit_datasets(sensitive_attr='sex', save_dir=SAVE_DIR)

    print("\n" + "="*65)
    print("Sample rows (action | reward | propensity | sensitive attrs)")
    print("="*65)

    for name, data in [
        ('Drug    (random)', student_data),
        ('Student (random)', student_data),
        ('Adult   (random)', adult_data),
        ('Loan    (random)', loan_data),
    ]:
        d = data['random']['train']
        sample = pd.concat([d['actions'], d['reward'], d['sensitive']], axis=1)
        print(f"\n── {name} ──")
        print(sample.head(3).to_string(index=False))
