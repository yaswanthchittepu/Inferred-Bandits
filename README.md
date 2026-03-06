# Group Fair Bandits — Code Explanation

## Overview

This codebase trains a contextual bandit policy on offline data while enforcing group fairness constraints. The disparity between two demographic groups (defined by a sensitive attribute) is measured either on IS-weighted rewards or IS-weighted actions. Several training modes are supported, ranging from unconstrained to statistically rigorous.

---

## Installation

```bash
conda create -n fair_bandits python=3.10
conda activate fair_bandits
pip install -r requirements.txt
```

---

## File Structure

```
optimize.py                               # Main training script (gradient-based)
optimize_cma.py                           # CMA-ES training script (derivative-free)
inferred-optimize-no-safe.py              # Gradient-based script with inferred (noisy) sensitive attributes
inferred-optimize-no-safe-cma.py          # CMA-ES script with inferred (noisy) sensitive attributes
policy.py                                 # PolicyNet definition
utils.py                                  # Constraint functions, safety test, test evaluation
configs/default.yaml                      # Default hyperparameters for optimize.py
configs/default_cma.yaml                  # Default hyperparameters for optimize_cma.py
configs/default_inferred_no_safe.yaml     # Default hyperparameters for inferred-optimize-no-safe.py
configs/default_inferred_no_safe_cma.yaml # Default hyperparameters for inferred-optimize-no-safe-cma.py
bandit_data_pipeline.py                   # Data preprocessing and dataset creation
```

---

## Data Format

Datasets are saved as a nested dict via joblib:

```
datasets[behavior_policy_type] = {
    'train':  {'context': DataFrame, 'actions': DataFrame, 'reward': Series, 'sensitive': Series},
    'safety': { ... },
    'test':   { ... }
}
```

- `context`: feature matrix (n_samples x n_features)
- `actions`: DataFrame with columns `action`, `propensity`, `true_label`
- `reward`: observed reward from the behavior policy's action
- `sensitive`: binary (0/1) group membership

---

## Policy (`policy.py`)

`PolicyNet` is a 2-layer MLP:

```
input_dim → Linear(256) → ReLU → Linear(256) → ReLU → Linear(n_actions) → Softmax
```

Outputs a probability distribution over actions for each context.

---

## Importance Sampling (IS)

Since data was collected by a behavior policy `pi_b`, we use IS to estimate what the learned policy `pi_theta` would achieve:

```
IS-reward  = (pi_theta(a|x) / pi_b(a|x)) * r
IS-action  = (pi_theta(a|x) / pi_b(a|x)) * a
```

The policy gradient objective maximizes the expected IS-weighted reward:

```
reward_obj_term = E[ IS-reward * log(pi_theta(a|x)) ]
```

---

## Training Modes (`optimize.py`)

All modes share the same data loading, policy architecture, and IS reward computation. They differ in whether and how fairness constraints are enforced.

### `naive`
Pure reward maximization. No fairness constraint. No dual variables.

```
loss = -reward_obj_term
```

### `seldonian`
Statistically rigorous Seldonian algorithm. Uses UCB-based fairness constraints derived from a t-test (Student's or Welch's). Two dual variables `lambda1`, `lambda2` are trained alongside the policy — one per direction of the disparity constraint (g1 > g2 and g2 > g1).

```
loss = -(reward_obj_term - constraint_term)
```

The constraint term is a differentiable surrogate using the policy gradient trick on the IS-weighted group metrics, augmented with a UCB correction scaled by the dual variables.

Dual variables are updated to maximize the Lagrangian:
```
lambda_loss = -((ucb_g1_v_g2 - epsilon) * lambda1) - ((ucb_g2_v_g1 - epsilon) * lambda2)
```

Dual variables are clamped to `log_lambda_max` in log space to prevent blow-up.

### `lagrange_exp`
Same Lagrangian structure as `seldonian`, but the constraint uses only the **expectation difference** — no UCB correction. Dual variables are trained the same way but updated using raw group mean differences instead of UCB bounds.

```
constraint_term = (lambda1 - lambda2) * (E_g1[metric * log pi] - E_g2[metric * log pi])
lambda_loss = -((mean_g1 - mean_g2 - epsilon) * lambda1) - ((mean_g2 - mean_g1 - epsilon) * lambda2)
```

### `fixed_lambda`
No dual variable training. A fixed scalar `fixed_lambda` (set in config) penalizes the signed expectation difference directly. The sign is determined by which group currently has a higher mean (detached — no gradient through sign).

```
sign = sign(mean_g1 - mean_g2)   # detached
constraint_term = fixed_lambda * sign * (E_g1[metric * log pi] - E_g2[metric * log pi])
loss = -(reward_obj_term - constraint_term)
```

---

## Constraint Functions (`utils.py`)

Both `students_ttest_ucb_constraint` and `welchs_ttest_ucb_constraint` share the same interface and support three modes via keyword arguments:

| kwarg | Behavior |
|---|---|
| default (`no_ucb=False, lagrange_exp=False`) | Full UCB constraint for `seldonian` |
| `lagrange_exp=True` | Expectation-only constraint with trainable dual vars |
| `no_ucb=True` | Expectation-only constraint with fixed scalar lambda |

**Student's t-test UCB** assumes equal group sizes and uses per-group t critical values K1, K2:
```
ucb_g1_v_g2 = mean_g1 - mean_g2 + K1*std_g1 + K2*std_g2
```

**Welch's t-test UCB** handles unequal group sizes/variances using the Satterthwaite degrees of freedom:
```
v = (s1²/n1 + s2²/n2)² / [(s1²/n1)²/(n1-1) + (s2²/n2)²/(n2-1)]
ucb_g1_v_g2 = mean_g1 - mean_g2 + t(1-delta, v) * sqrt(s1²/n1 + s2²/n2)
```

**`bound_prop_ucb_constraint_no_safety`** is used exclusively by `inferred-optimize-no-safe.py`. It accounts for the fact that some training samples have inferred (noisy) sensitive attributes. Instead of a t-test UCB, it uses bound propagation through a confusion matrix `M` to correct for classifier errors. See the [Inferred Sensitive Attributes](#inferred-sensitive-attributes-inferred-optimize-no-safepy) section for details.

---

## Safety Test and Test Evaluation (`utils.py`)

After training, two checks are run:

**`safety_test`**: Evaluates the policy on the held-out safety split. Uses the same UCB bounds (Student's or Welch's) to check with high probability that the disparity constraint is satisfied. Returns `'Safe'` or `'NSF'` (No Solution Found).

**`evaluate_on_test`**: Evaluates on the test split using a plain absolute difference (no UCB). Returns `'Safe'` or `'Unsafe'`.

Both functions share `gather_is_rewards`, which runs the policy over a dataset split and returns IS-weighted rewards, IS-weighted actions, and sensitive group masks.

---

## Configuration

All hyperparameters live in `configs/default.yaml`. CLI args override YAML values when provided:

```bash
python optimize.py --config configs/default.yaml --mode seldonian --seed 0
```

| Parameter | Description |
|---|---|
| `mode` | `seldonian`, `naive`, `fixed_lambda`, `lagrange_exp` |
| `behavior_policy_type` | Behavior policy used to collect the offline data: `random` (uniform over all actions), `tweak1` (biased — action 0 chosen with prob ρ=0.9, remaining actions share 0.1), `mixed` (logistic regression fit on training labels — realistic but suboptimal. For student loan dataset, we use a simple rule based policy that assigns a loan if they had paid their previous credits and no otherwise) |
| `constraint_type` | `students-ttest`, `welchs-ttest` |
| `disparity_type` | `reward` (IS-reward gap) or `action` (IS-action gap) |
| `epsilon` | Maximum allowed group disparity |
| `fail_prob` | Target failure probability for UCB bounds |
| `fixed_lambda` | Penalty weight for `fixed_lambda` mode |
| `dual_opt_lr` | Learning rate for dual variable optimizer |
| `log_lambda_max` | Clamp on log-space dual variables |

---

## Inferred Sensitive Attributes (`inferred-optimize-no-safe.py`)

This script handles the realistic setting where the sensitive attribute is **not fully observed** for all training samples. A portion of training data uses a **classifier-inferred** (and therefore noisy) sensitive attribute, while the rest has ground-truth labels.

### Partition Split

At startup, the training data is split into two disjoint partitions:

- **Ground-truth partition** (`1 - inferred_proportion`): sensitive attribute is accurate.
- **Inferred partition** (`inferred_proportion`): sensitive attribute is corrupted by the classifier.

```python
n_inferred   = int(n_train * INFERRED_PROPORTION)
inferred_idx = np.random.choice(n_train, size=n_inferred, replace=False)
```

### Asymmetric Label Noise (FPR / FNR)

The inferred partition's sensitive attribute is corrupted using asymmetric error rates:

| Rate | Meaning |
|---|---|
| `fpr` (α) | P(inferred=1 \| true=0) — false positive rate |
| `fnr` (β) | P(inferred=0 \| true=1) — false negative rate |

When `fpr == fnr`, this reduces to symmetric noise. Flipping is applied independently per sample:

```python
flip_probs = np.where(inferred_orig == 0, FPR, FNR)
flip_mask  = np.random.random(n_inferred) < flip_probs
sensitive_noisy[inferred_idx[flip_mask]] ^= 1
```

### Group Size Counting

Group sizes (used in the UCB constraint) are computed **from the ground-truth partition only**, since inferred labels are unreliable:

```python
gt_sensitive     = sensitive_noisy[~is_inferred]
train_grp_1_size = int(gt_sensitive.sum())       # sens=1
train_grp_2_size = int((~is_inferred).sum()) - train_grp_1_size  # sens=0
```

### Constraint: Bound Propagation via Confusion Matrix

`bound_prop_ucb_constraint_no_safety` (from `utils.py`) corrects for classifier noise by propagating bounds through the confusion matrix **M**, where:

```
M[s, s'] = P(S=s | S'=s')    (posterior prob using Bayes rule with FPR, FNR, group priors)
```

Row sums of M must equal 1 (asserted at runtime). The inverse `M_inv` is used to "undo" the label noise and obtain corrected group-level estimates.

The function takes an extra `inferred_mask` tensor (shape `(B,)`, bool) indicating which batch samples are from the inferred partition.

### Training Modes

Only two modes are supported:

| Mode | Description |
|---|---|
| `naive` | Unconstrained IS-reward maximization |
| `seldonian` | Lagrangian fairness with bound-propagation UCB constraint |

### Usage

```bash
python inferred-optimize-no-safe.py --config configs/default_inferred_no_safe.yaml --mode seldonian --seed 0
```

| Parameter | Description |
|---|---|
| `inferred_proportion` | Fraction of training data using inferred (noisy) sensitive attr |
| `fpr` | False positive rate of the sensitive attribute classifier (α) |
| `fnr` | False negative rate of the sensitive attribute classifier (β) |

All other parameters (`mode`, `epsilon`, `fail_prob`, etc.) are identical to `optimize.py`.

### Outputs

Saved to `{output_dir}/{dataset}/{sensitive_attr}/{constraint_trunc}/{behavior_policy}/`:

- `policy.pt` — trained policy weights
- `results.txt` — seed, safety_flag, test_flag, verdict

> **Note:** Safety test and test evaluation still use the ground-truth sensitive attribute (safety and test splits are assumed to have clean labels).

---

## CMA-ES Optimization (`optimize_cma.py`)

An alternative to the gradient-based approach using **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy), a derivative-free black-box optimizer.

### Policy

`LinearPolicy` is a pure numpy softmax policy (no PyTorch autograd):
```
logits = x @ W.T + b    →    softmax    →    action probs
```
Parameters are a flat vector `[W, b]` of size `input_dim * n_actions + n_actions`.

### Objective

For each candidate parameter vector, iterate over training data and compute the mean IS-weighted reward. CMA-ES **minimizes**, so the objective returned is the **negated** mean IS-reward.

### Constraint

Uses `bound_propagation` (from `utils.py`) to compute a UCB on `|E_g1 - E_g2|`. If the UCB exceeds `epsilon`, the solution is assigned a hard barrier penalty (`1e6`). Otherwise the IS-return objective is used.

```
if ucb_abs_diff > epsilon:   fitness = 1e6        # infeasible
else:                        fitness = -mean_is_return
```

### Parallelism

Candidate solutions within each generation are evaluated in parallel using `joblib.Parallel`. Control with `n_jobs` (default: `-1` = all CPUs).

### Usage

```bash
python optimize_cma.py --config configs/default_cma.yaml
```

| Parameter | Description |
|---|---|
| `sigma0` | Initial step size for CMA-ES |
| `maxiter` | Maximum number of CMA-ES generations |
| `popsize` | Population size per generation (null = CMA-ES default: `4 + floor(3*ln(N))`) |
| `n_jobs` | Number of parallel workers for solution evaluation |

### Outputs

Saved to `{output_dir}/{dataset}/{sensitive_attr}/{constraint_trunc}/{behavior_policy}/cma/`:

- `best_params.npy` — best found parameter vector
- `results.txt` — seed, safety_flag, test_flag, verdict

---

## Inferred CMA-ES (`inferred-optimize-no-safe-cma.py`)

Combines the CMA-ES derivative-free optimizer from `optimize_cma.py` with the inferred sensitive attribute handling from `inferred-optimize-no-safe.py`. Useful when you want the black-box optimizer but training data has partially noisy group labels.

### Policy

Same `LinearPolicy` (numpy softmax) as `optimize_cma.py`. No PyTorch autograd required.

### Inferred Partition

Identical split and flipping logic as `inferred-optimize-no-safe.py`: `inferred_proportion` of training data has its sensitive attribute corrupted with asymmetric FPR/FNR noise.

### Barrier Function (`inferred_bound_barrier`)

Unlike `optimize_cma.py` which uses `bound_propagation` directly, this script uses `inferred_bound_barrier` — a dedicated function that accounts for label noise in the inferred partition.

It computes the M-matrix (confusion matrix of posterior probabilities) and its inverse, then constructs separate UCBs for the ground-truth and inferred partitions:

```
E[X | S=1] ≈ p1 * E[X|S=1, ground]  +  p2 * E[X|S'=1, inferred]  +  p3 * E[X|S'=0, inferred]
```

where `p2`, `p3` are weighted by `M_inv` rows to correct for label noise. A t-test UCB is applied only on the ground-truth term (where labels are reliable); inferred means are used without a UCB correction.

```
if ucb_abs_diff > epsilon:   fitness = 1e6        # infeasible
else:                        fitness = -mean_is_reward
```

### Usage

```bash
python inferred-optimize-no-safe-cma.py --config configs/default_inferred_no_safe_cma.yaml
```

All CMA-ES parameters (`sigma0`, `maxiter`, `popsize`, `n_jobs`, `eval_freq`) and inferred parameters (`inferred_proportion`, `fpr`, `fnr`) are supported. See `configs/default_inferred_no_safe_cma.yaml`.

### Outputs

Saved to `{output_dir}/{dataset}/{sensitive_attr}/{constraint_trunc}/{behavior_policy}/inferred_cma/`:

- `best_params.npy` — best found parameter vector
- `results.txt` — seed, safety_flag, test_flag, verdict

> **Note:** Safety and test splits use ground-truth sensitive attributes (clean labels assumed).

---

## Outputs (`optimize.py`)

Saved to `{output_dir}/{dataset}/{sensitive_attr}/{constraint_trunc}/{behavior_policy}/`:

- `policy.pt` — trained policy weights (`torch.save`)
- `results.txt` — seed, safety_flag, test_flag, verdict
