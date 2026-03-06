import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
import os, sys, ipdb
import argparse
import yaml
import wandb
import cma
import torch
from utils import bound_propagation
from scipy import stats
import math


class LinearPolicy:
    def __init__(self, input_dim, n_actions):
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.param_dim = input_dim * n_actions + n_actions  # W + b

    def get_probs(self, x, params):
        """x: (B, D) numpy array, params: 1D numpy array. Returns (B, n_actions) numpy array."""
        W = params[:self.input_dim * self.n_actions].reshape(self.n_actions, self.input_dim)
        b = params[self.input_dim * self.n_actions:]
        logits = x @ W.T + b  # (B, n_actions)
        logits = logits - logits.max(axis=1, keepdims=True)  # numerical stability
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def get_action_probs(self, x, params, actions):
        """Returns pi(a|x) for each (x, a) pair. actions: (B,) int array."""
        probs = self.get_probs(x, params)  # (B, n_actions)
        return probs[np.arange(len(actions)), actions]  # (B,)


def eval_metrics(params, train_X, train_a, train_pi_b, train_r, train_s, train_true_labels,
                 policy, batch_size, constraint_kwargs):
    """Full evaluation of params on training data. Returns IS and on-policy metrics."""
    all_is_rewards, all_is_actions, all_pi_a, all_sensitive, all_pred_actions = [], [], [], [], []
    for b_sidx in range(0, len(train_X), batch_size):
        batch_X    = train_X   [b_sidx : b_sidx + batch_size]
        batch_a    = train_a   [b_sidx : b_sidx + batch_size]
        batch_pi_b = train_pi_b[b_sidx : b_sidx + batch_size]
        batch_r    = train_r   [b_sidx : b_sidx + batch_size]
        batch_s    = train_s   [b_sidx : b_sidx + batch_size]
        probs = policy.get_probs(batch_X, params)
        pi_a  = probs[np.arange(len(batch_a)), batch_a]
        all_is_rewards.append((pi_a / (batch_pi_b + 1e-8)) * batch_r)
        all_is_actions.append((pi_a / (batch_pi_b + 1e-8)) * batch_a.astype(float))
        all_pi_a.append(pi_a)
        all_sensitive.append(batch_s)
        all_pred_actions.append(probs.argmax(axis=1))
    is_r_np   = np.concatenate(all_is_rewards)
    is_a_np   = np.concatenate(all_is_actions)
    pi_a_np   = np.concatenate(all_pi_a)
    s_np      = np.concatenate(all_sensitive).ravel().astype(bool)
    pred_a_np = np.concatenate(all_pred_actions)
    metric  = is_r_np if constraint_kwargs['disparity_type'] == 'reward' else is_a_np
    mean_g1 = metric[s_np].mean()
    mean_g2 = metric[~s_np].mean()
    is_r_t = torch.tensor(is_r_np, dtype=torch.float32).unsqueeze(1)
    is_a_t = torch.tensor(is_a_np, dtype=torch.float32).unsqueeze(1)
    pi_a_t = torch.tensor(pi_a_np, dtype=torch.float32).unsqueeze(1)
    s_t    = torch.tensor(s_np,    dtype=torch.bool)
    _, ucb_abs_diff = bound_propagation(is_r_t, is_a_t, pi_a_t, s_t, constraint_kwargs)
    match        = pred_a_np == train_true_labels
    rewards_eval = np.where(match, 1.0, -1.0)
    return {
        'mean_is_reward':   is_r_np.mean(),
        'mean_g1':          mean_g1,
        'mean_g2':          mean_g2,
        'abs_diff':         abs(mean_g1 - mean_g2),
        'ucb_abs_diff':     ucb_abs_diff.item(),
        'on_policy_reward': rewards_eval.mean(),
        'on_policy_g1':     rewards_eval[s_np].mean()  if s_np.sum()    > 0 else float('nan'),
        'on_policy_g2':     rewards_eval[~s_np].mean() if (~s_np).sum() > 0 else float('nan'),
    }


# ---------------------------------------------------------------------------
# PLACEHOLDER: inferred-aware feasibility check for CMA-ES barrier
# ---------------------------------------------------------------------------
def inferred_bound_barrier(is_reward, is_action, pi_a, sensitive_mask, inferred_mask, constraint_kwargs, epsilon):
    """
    Inferred-aware feasibility check for the CMA-ES barrier function.

    PLACEHOLDER — implement the inferred-aware bound propagation here using
    FPR, FNR, and M_inv to correct for label noise in the inferred partition.

    Parameters
    ----------
    is_rewards   : (N,) numpy array — IS-weighted rewards
    is_actions   : (N,) numpy array — IS-weighted actions
    pi_a         : (N,) numpy array — pi(a|x) probabilities
    sensitive    : (N,) numpy array — (noisy) sensitive attribute labels (0/1)
    is_inferred  : (N,) bool array  — True for samples from the inferred partition
    constraint_kwargs : dict        — contains fpr, fnr, grp counts, epsilon, fail_prob, etc.
    epsilon      : float            — maximum allowed disparity

    Returns
    -------
    feasible : bool — True if the constraint is satisfied (policy is safe)
    """
    # TODO: implement inferred-aware bound propagation
    # Constraint Terms
    if(constraint_kwargs['disparity_type'] == 'reward'):
        group_1_is_metric = is_reward[sensitive_mask]
        group_2_is_metric = is_reward[~sensitive_mask]
    elif(constraint_kwargs['disparity_type'] == 'action'):
        group_1_is_metric = is_action[sensitive_mask]
        group_2_is_metric = is_action[~sensitive_mask]
    else:
        raise ValueError(f"Invalid disparity type provided: {constraint_kwargs['disparity_type']}")

    # FPR for 0s, FNR for 1s
    # Grp 1 corresponds to sensitive attribute = 1
    # Grp 2 corresponds to sensitive attribute = 0

    prob_grp1 = constraint_kwargs['grp1_cnt_train_data']/(constraint_kwargs['grp1_cnt_train_data'] + constraint_kwargs['grp2_cnt_train_data'])
    prob_grp2 = 1 - prob_grp1

    # m11 = P(S=1|S'=1) = P(S'=1 | S = 1) P(S=1) / ( P(S'=1 | S = 1) P(S=1) +  P(S'= 1 | S = 0) P(S=0))
    m11 = ((1-constraint_kwargs['fnr'])*prob_grp1)/(((1-constraint_kwargs['fnr'])*prob_grp1) + ((constraint_kwargs['fpr'])*prob_grp2))

    # m12 = P(S=0|S'=1) = P(S'=1|S=0) P(S=0) / (P(S'=1|S=0) P(S=0) + P(S'=1|S=1) P(S=1))
    m12 = ((constraint_kwargs['fpr'])*prob_grp2)/(((constraint_kwargs['fpr'])*prob_grp2) + ((1-constraint_kwargs['fnr'])*prob_grp1))


    # m21 = P(S=1 | S'=0) = P(S'=0 | S =1) P(S=1) / (P(S'=0 | S =1) P(S=1) + P(S'=0 | S =0) P(S=0))
    m21 = ((constraint_kwargs['fnr'])*prob_grp1)/(((constraint_kwargs['fnr'])*prob_grp1) + ((1-constraint_kwargs['fpr'])*prob_grp2))

    # m22 = P(S=0 | S'=0) = P(S'=0 | S = 0) P(S=0) / (P(S'=0 | S = 0) P(S=0) + P(S'=0 | S = 1) P(S=1))
    m22 = ((1-constraint_kwargs['fpr'])*prob_grp2)/(((1-constraint_kwargs['fpr'])*prob_grp2) + ((constraint_kwargs['fnr'])*prob_grp1))
    
    M = np.array([[m11, m12], [m21, m22]])
    assert np.allclose(M.sum(axis=1), 1.0), f"M rows do not sum to 1: {M.sum(axis=1)}"
    M_inv = np.linalg.inv(M)

    # mu' = [E[X|S'=1, inferred], E[X|S'=0, inferred]]
    # mu = [E[X|S=1, inferred], E[X|S=0, inferred]]
    # mu = M.inv() mu'

    group_1_is_metric_ground = group_1_is_metric[~inferred_mask]
    group_1_is_metric_inferred = group_1_is_metric[inferred_mask]
    group_2_is_metric_ground = group_2_is_metric[~inferred_mask]
    group_2_is_metric_inferred = group_2_is_metric[inferred_mask]

    group_1_is_metric_ground_mean, group_1_is_metric_ground_std = group_1_is_metric_ground.mean(), group_1_is_metric_ground.std(ddof=1)


    # Compute E[X | S=1]
    # Coefficient for E[X|S=1, ground]
    p1_grp1 = (constraint_kwargs['ground_truth_train_data_size']/(constraint_kwargs['ground_truth_train_data_size'] + constraint_kwargs['inferred_train_data_size']))
    p2_grp1 = (1-p1_grp1)*M_inv[0,0] # Coefficient for E[X|S'=1, inferred]
    p3_grp1 = (1-p1_grp1)*M_inv[0,1] # Coefficient for E[X|S'=0, inferred]

    K1 = stats.t.ppf(1 - (constraint_kwargs['fail_prob']/4.0), constraint_kwargs['safety_data_grp_1_size']-1)/math.sqrt(constraint_kwargs['safety_data_grp_2_size'])
    lcb_p1_grp1 = p1_grp1*(group_1_is_metric_ground.mean() - K1*group_1_is_metric_ground.std(ddof=1))
    ucb_p1_grp1 = p1_grp1*(group_1_is_metric_ground.mean() + K1*group_1_is_metric_ground.std(ddof=1))
    lcb_p1_grp1 += (p2_grp1*group_1_is_metric_inferred.mean()) + (p3_grp1*group_2_is_metric_inferred.mean())
    ucb_p1_grp1 += (p2_grp1*group_1_is_metric_inferred.mean()) + (p3_grp1*group_2_is_metric_inferred.mean())

    # COmpute E[X | S=0]
    # Coefficient for E[X|S=0, ground]
    p1_grp2 = (constraint_kwargs['ground_truth_train_data_size']/(constraint_kwargs['ground_truth_train_data_size'] + constraint_kwargs['inferred_train_data_size']))
    p2_grp2 = (1-p1_grp2)*M_inv[1,0] # Coefficient for E[X|S'=1, inferred]
    p3_grp2 = (1-p1_grp2)*M_inv[1,1] # Coefficient for E[X|S'=0, inferred]

    K2 = stats.t.ppf(1 - (constraint_kwargs['fail_prob']/4.0), constraint_kwargs['safety_data_grp_2_size']-1)/math.sqrt(constraint_kwargs['safety_data_grp_2_size'])
    lcb_p1_grp2 = p1_grp2*(group_2_is_metric_ground.mean() - K2*group_2_is_metric_ground.std(ddof=1))
    ucb_p1_grp2 = p1_grp2*(group_2_is_metric_ground.mean() + K2*group_2_is_metric_ground.std(ddof=1))
    lcb_p1_grp2 += (p2_grp2*group_1_is_metric_inferred.mean()) + (p3_grp2*group_2_is_metric_inferred.mean())
    ucb_p1_grp2 += (p2_grp2*group_1_is_metric_inferred.mean()) + (p3_grp2*group_2_is_metric_inferred.mean()) 

    ucb_diff = ucb_p1_grp1-lcb_p1_grp2
    lcb_diff = lcb_p1_grp1-ucb_p1_grp2

    ucb_abs_diff = max(abs(lcb_diff), abs(ucb_diff))
    lcb_abs_diff = 0.0 if (lcb_diff <= 0 and ucb_diff >= 0) else min(abs(lcb_diff), abs(ucb_diff))

    return True if(ucb_abs_diff <= constraint_kwargs['epsilon']) else False

    # raise NotImplementedError("inferred_bound_barrier is not yet implemented")
# ---------------------------------------------------------------------------


def evaluate_solution(params, train_X, train_a, train_pi_b, train_r, train_s, train_is_inferred,
                      policy, batch_size, constraint_kwargs, epsilon):
    """Compute the CMA-ES fitness for a candidate parameter vector."""
    all_is_rewards, all_is_actions, all_pi_a, all_sensitive, all_is_inferred = [], [], [], [], []
    for b_sidx in range(0, len(train_X), batch_size):
        batch_X          = train_X          [b_sidx : b_sidx + batch_size]
        batch_a          = train_a          [b_sidx : b_sidx + batch_size]
        batch_pi_b       = train_pi_b       [b_sidx : b_sidx + batch_size]
        batch_r          = train_r          [b_sidx : b_sidx + batch_size]
        batch_s          = train_s          [b_sidx : b_sidx + batch_size]
        batch_is_inf     = train_is_inferred[b_sidx : b_sidx + batch_size]
        pi_a             = policy.get_action_probs(batch_X, params, batch_a)
        all_is_rewards.append((pi_a / (batch_pi_b + 1e-8)) * batch_r)
        all_is_actions.append((pi_a / (batch_pi_b + 1e-8)) * batch_a.astype(float))
        all_pi_a.append(pi_a)
        all_sensitive.append(batch_s)
        all_is_inferred.append(batch_is_inf)

    is_rewards_np  = np.concatenate(all_is_rewards)
    is_actions_np  = np.concatenate(all_is_actions)
    pi_a_np        = np.concatenate(all_pi_a)
    sensitive_np   = np.concatenate(all_sensitive)
    is_inferred_np = np.concatenate(all_is_inferred)

    feasible = inferred_bound_barrier(
        is_rewards_np, is_actions_np, pi_a_np, sensitive_np, is_inferred_np,
        constraint_kwargs, epsilon)
    if not feasible:
        return 1e6
    return -is_rewards_np.mean()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',               type=str,   default='configs/default_inferred_no_safe_cma.yaml')
    parser.add_argument('--seed',                 type=int,   default=None)
    parser.add_argument('--output_dir',           type=str,   default=None)
    parser.add_argument('--dataset_name',         type=str,   default=None)
    parser.add_argument('--sensitive_attribute',  type=str,   default=None)
    parser.add_argument('--behavior_policy_type', type=str,   default=None)
    parser.add_argument('--bandit_data_dir',      type=str,   default=None)
    parser.add_argument('--constraint_type',      type=str,   default=None)
    parser.add_argument('--disparity_type',       type=str,   default=None)
    parser.add_argument('--epsilon',              type=float, default=None)
    parser.add_argument('--fail_prob',            type=float, default=None)
    parser.add_argument('--sigma0',               type=float, default=None)
    parser.add_argument('--maxiter',              type=int,   default=None)
    parser.add_argument('--popsize',              type=int,   default=None)
    parser.add_argument('--batch_size',           type=int,   default=None)
    parser.add_argument('--n_jobs',               type=int,   default=None)
    parser.add_argument('--eval_freq',            type=int,   default=None)
    parser.add_argument('--inferred_proportion',  type=float, default=None)
    parser.add_argument('--fpr',                  type=float, default=None)
    parser.add_argument('--fnr',                  type=float, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for key, val in vars(args).items():
        if key != 'config' and val is not None:
            cfg[key] = val

    DATSET_NAME          = cfg['dataset_name']
    SENSITIVE_ATTRIBUTE  = cfg['sensitive_attribute']
    BEHAVIOR_POLICY_TYPE = cfg['behavior_policy_type']
    BANDIT_DATA_PATH     = os.path.join(cfg['bandit_data_dir'], DATSET_NAME, SENSITIVE_ATTRIBUTE, 'datasets.joblib')
    CONSTRAINT_TYPE      = cfg['constraint_type']
    DISPARITY_TYPE       = cfg['disparity_type']
    EPSILON              = cfg['epsilon']
    FAIL_PROB            = cfg['fail_prob']
    SIGMA0               = cfg['sigma0']
    MAXITER              = cfg['maxiter']
    POPSIZE              = cfg.get('popsize', None)
    BATCH_SIZE           = cfg['batch_size']
    N_JOBS               = cfg.get('n_jobs', -1)
    EVAL_FREQ            = cfg['eval_freq']
    INFERRED_PROPORTION  = cfg['inferred_proportion']
    FPR                  = cfg['fpr']   # alpha: P(inferred=1 | true=0)
    FNR                  = cfg['fnr']   # beta:  P(inferred=0 | true=1)
    CONSTRAINT_TRUNC     = CONSTRAINT_TYPE.split('-')[0]
    OUTPUT_DIR           = os.path.join(cfg['output_dir'], DATSET_NAME, SENSITIVE_ATTRIBUTE,
                                        CONSTRAINT_TRUNC, BEHAVIOR_POLICY_TYPE, 'inferred_cma')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    SEED                 = cfg['seed']

    np.random.seed(SEED)

    assert BEHAVIOR_POLICY_TYPE in ['random', 'tweak1', 'mixed']
    assert CONSTRAINT_TYPE in ['students-ttest', 'welchs-ttest']
    assert DISPARITY_TYPE in ['reward', 'action']

    wandb.init(
        project="group-fair-bandits",
        name=f"{DATSET_NAME}-{SENSITIVE_ATTRIBUTE}-inferred-cma-{CONSTRAINT_TRUNC}-{BEHAVIOR_POLICY_TYPE}",
        config={
            "dataset_name":         DATSET_NAME,
            "run_mode":             "inferred_cma",
            "sensitive_attribute":  SENSITIVE_ATTRIBUTE,
            "behavior_policy_type": BEHAVIOR_POLICY_TYPE,
            "constraint_type":      CONSTRAINT_TYPE,
            "disparity_type":       DISPARITY_TYPE,
            "epsilon":              EPSILON,
            "fail_prob":            FAIL_PROB,
            "sigma0":               SIGMA0,
            "maxiter":              MAXITER,
            "popsize":              POPSIZE,
            "eval_freq":            EVAL_FREQ,
            "inferred_proportion":  INFERRED_PROPORTION,
            "FPR":                  FPR,
            "FNR":                  FNR,
        },
    )

    datasets = joblib.load(BANDIT_DATA_PATH)
    data        = datasets[BEHAVIOR_POLICY_TYPE]
    train_data  = data['train']
    safety_data = data['safety']
    test_data   = data['test']

    # Split train_data into ground-truth and inferred partitions.
    # The inferred partition has its sensitive attribute corrupted at FPR / FNR rates.
    n_train      = len(train_data['context'])
    n_inferred   = int(n_train * INFERRED_PROPORTION)
    inferred_idx = np.random.choice(n_train, size=n_inferred, replace=False)
    print(f"Train split — ground-truth: {n_train - n_inferred}, inferred: {n_inferred} "
          f"(proportion={INFERRED_PROPORTION}, fpr={FPR}, fnr={FNR})")

    sensitive_noisy = train_data['sensitive'].values.ravel().copy().astype(int)
    inferred_orig   = sensitive_noisy[inferred_idx]          # true labels before flipping
    flip_probs      = np.where(inferred_orig == 0, FPR, FNR) # FPR for 0s, FNR for 1s
    flip_mask       = np.random.random(n_inferred) < flip_probs
    sensitive_noisy[inferred_idx[flip_mask]] = 1 - sensitive_noisy[inferred_idx[flip_mask]]
    n_fp = int(flip_mask[inferred_orig == 0].sum())
    n_fn = int(flip_mask[inferred_orig == 1].sum())
    print(f"Inferred flips — FP (0→1): {n_fp}, FN (1→0): {n_fn}, total: {flip_mask.sum()}")

    is_inferred = np.zeros(n_train, dtype=bool)
    is_inferred[inferred_idx] = True

    # Group sizes counted on ground-truth partition only (sensitive=1 → grp1, sensitive=0 → grp2)
    gt_sensitive     = sensitive_noisy[~is_inferred]
    train_grp_1_size = int(gt_sensitive.sum())
    train_grp_2_size = int((~is_inferred).sum()) - train_grp_1_size
    print(f"Ground-truth partition — grp1 (sens=1): {train_grp_1_size}, grp2 (sens=0): {train_grp_2_size}")

    input_dim = train_data['context'].shape[1]
    n_actions = max(train_data['actions']['true_label'].max(),
                    train_data['actions']['action'].max()) + 1

    train_X           = train_data['context'].values
    train_a           = train_data['actions']['action'].values
    train_pi_b        = train_data['actions']['propensity'].values
    train_r           = train_data['reward'].values
    train_s           = sensitive_noisy          # noisy sensitive attributes (inferred partition flipped)
    train_is_inferred = is_inferred
    train_true_labels = train_data['actions']['true_label'].values

    policy = LinearPolicy(input_dim, n_actions)
    x0     = np.zeros(policy.param_dim)

    safety_data_size       = len(safety_data['context'])
    safety_data_grp_1_size = safety_data['sensitive'].values.sum().item()
    safety_data_grp_2_size = safety_data_size - safety_data_grp_1_size

    constraint_kwargs = {
        'fail_prob':                    FAIL_PROB,
        'epsilon':                      EPSILON,
        'safety_data_grp_1_size':       safety_data_grp_1_size,
        'safety_data_grp_2_size':       safety_data_grp_2_size,
        'constraint_type':              CONSTRAINT_TYPE,
        'disparity_type':               DISPARITY_TYPE,
        'inferred_train_data_size':     n_inferred,
        'ground_truth_train_data_size': n_train - n_inferred,
        'grp1_cnt_train_data':          train_grp_1_size,
        'grp2_cnt_train_data':          train_grp_2_size,
        'fpr':                          FPR,
        'fnr':                          FNR,
    }

    cma_opts = {'seed': SEED, 'maxiter': MAXITER, 'verbose': 1, 'tolfun': 0, 'tolx': 0}
    if POPSIZE is not None:
        cma_opts['popsize'] = POPSIZE

    generation = 0
    es = cma.CMAEvolutionStrategy(x0, SIGMA0, cma_opts)
    while not es.stop():
        solutions = es.ask()
        fitnesses = Parallel(n_jobs=N_JOBS)(
            delayed(evaluate_solution)(
                params, train_X, train_a, train_pi_b, train_r, train_s, train_is_inferred,
                policy, BATCH_SIZE, constraint_kwargs, EPSILON)
            for params in solutions
        )
        es.tell(solutions, fitnesses)
        es.disp()

        feasible = [f for f in fitnesses if f < 1e5]
        log_dict = {
            'train/n_feasible': len(feasible),
            'train/fbest': -es.result.fbest if es.result.fbest < 1e5 else float('nan'),
        }
        if generation % EVAL_FREQ == 0:
            m = eval_metrics(es.result.xbest, train_X, train_a, train_pi_b, train_r, train_s,
                             train_true_labels, policy, BATCH_SIZE, constraint_kwargs)
            log_dict.update({
                'eval/mean_is_reward':   m['mean_is_reward'],
                'eval/mean_g1':          m['mean_g1'],
                'eval/mean_g2':          m['mean_g2'],
                'eval/abs_diff':         m['abs_diff'],
                'eval/ucb_abs_diff':     m['ucb_abs_diff'],
                'eval/on_policy_reward': m['on_policy_reward'],
                'eval/on_policy_g1':     m['on_policy_g1'],
                'eval/on_policy_g2':     m['on_policy_g2'],
            })
        wandb.log(log_dict, step=generation)
        generation += 1

    best_params = es.result.xbest
    np.save(os.path.join(OUTPUT_DIR, 'best_params.npy'), best_params)
    print(f"Best params norm: {np.linalg.norm(best_params):.4f}")

    # --- Safety Test & Test Evaluation ---
    # Safety and test splits are assumed to have clean (ground-truth) sensitive attributes.
    def gather_is(data):
        X    = data['context'].values
        a    = data['actions']['action'].values
        pi_b = data['actions']['propensity'].values
        r    = data['reward'].values
        s    = data['sensitive'].values
        pi_a = policy.get_action_probs(X, best_params, a)
        is_r   = torch.tensor((pi_a / (pi_b + 1e-8)) * r,               dtype=torch.float32).unsqueeze(1)
        is_a   = torch.tensor((pi_a / (pi_b + 1e-8)) * a.astype(float), dtype=torch.float32).unsqueeze(1)
        pi_a_t = torch.tensor(pi_a,                                      dtype=torch.float32).unsqueeze(1)
        sens   = torch.tensor(s.ravel().astype(bool),                    dtype=torch.bool)
        return is_r, is_a, pi_a_t, sens

    is_r, is_a, pi_a_t, sens = gather_is(safety_data)
    _, ucb_abs_diff = bound_propagation(is_r, is_a, pi_a_t, sens, constraint_kwargs)
    safety_flag = 'Safe' if ucb_abs_diff.item() < EPSILON else 'NSF'

    is_r, is_a, pi_a_t, sens = gather_is(test_data)
    m = is_r if DISPARITY_TYPE == 'reward' else is_a
    test_flag = 'Safe' if torch.abs(m[sens].mean() - m[~sens].mean()).item() < EPSILON else 'Unsafe'

    verdict = (f"Error: Our High Confidence Algorithm failed, Safety flag: {safety_flag}, Test Flag: {test_flag}"
               if safety_flag == 'Safe' and test_flag == 'Unsafe'
               else f"Success: Our High Confidence Algorithm succeeded, Safety flag: {safety_flag}, Test Flag: {test_flag}")
    print(verdict)

    results_path = os.path.join(OUTPUT_DIR, 'results.txt')
    with open(results_path, 'w') as f:
        f.write(f"seed:        {SEED}\n")
        f.write(f"safety_flag: {safety_flag}\n")
        f.write(f"test_flag:   {test_flag}\n")
        f.write(f"verdict:     {verdict}\n")
    print(f"Saved results → {results_path}")
