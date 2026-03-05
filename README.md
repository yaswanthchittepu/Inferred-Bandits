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
optimize.py             # Main training script (gradient-based)
optimize_cma.py         # CMA-ES training script (derivative-free)
policy.py               # PolicyNet definition
utils.py                # Constraint functions, safety test, test evaluation
configs/default.yaml    # Default hyperparameters for optimize.py
configs/default_cma.yaml # Default hyperparameters for optimize_cma.py
bandit_data_pipeline.py # Data preprocessing and dataset creation
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

## Outputs (`optimize.py`)

Saved to `{output_dir}/{dataset}/{sensitive_attr}/{constraint_trunc}/{behavior_policy}/`:

- `policy.pt` — trained policy weights (`torch.save`)
- `results.txt` — seed, safety_flag, test_flag, verdict
