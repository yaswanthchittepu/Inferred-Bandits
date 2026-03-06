import numpy as np
import joblib
from joblib import Parallel, delayed
import os, sys, ipdb
import json
import math
from scipy import stats
import argparse
import yaml
import wandb
import cma
import torch
from utils import bound_propagation


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


def evaluate_solution(params, train_X, train_a, train_pi_b, train_r, train_s,
                      policy, batch_size, constraint_kwargs, epsilon):
    all_is_rewards, all_is_actions, all_pi_a, all_sensitive = [], [], [], []
    for b_sidx in range(0, len(train_X), batch_size):
        batch_X    = train_X   [b_sidx : b_sidx + batch_size]
        batch_a    = train_a   [b_sidx : b_sidx + batch_size]
        batch_pi_b = train_pi_b[b_sidx : b_sidx + batch_size]
        batch_r    = train_r   [b_sidx : b_sidx + batch_size]
        batch_s    = train_s   [b_sidx : b_sidx + batch_size]
        pi_a       = policy.get_action_probs(batch_X, params, batch_a)
        all_is_rewards.append((pi_a / (batch_pi_b + 1e-8)) * batch_r)
        all_is_actions.append((pi_a / (batch_pi_b + 1e-8)) * batch_a.astype(float))
        all_pi_a.append(pi_a)
        all_sensitive.append(batch_s)

    all_is_rewards_t = torch.tensor(np.concatenate(all_is_rewards), dtype=torch.float32).unsqueeze(1)
    all_is_actions_t = torch.tensor(np.concatenate(all_is_actions), dtype=torch.float32).unsqueeze(1)
    all_pi_a_t       = torch.tensor(np.concatenate(all_pi_a),       dtype=torch.float32).unsqueeze(1)
    all_sensitive_t  = torch.tensor(np.concatenate(all_sensitive),  dtype=torch.bool)

    _, ucb_abs_diff = bound_propagation(all_is_rewards_t, all_is_actions_t, all_pi_a_t, all_sensitive_t, constraint_kwargs)
    if ucb_abs_diff.item() > epsilon:
        return 1e6
    return -all_is_rewards_t.mean().item()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',               type=str,   default='configs/default_cma.yaml')
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
    parser.add_argument('--train_frac',           type=float, default=None)
    parser.add_argument('--max_ex_train',         type=int,   default=None)
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
    TRAIN_FRAC           = cfg.get('train_frac', 1.0)
    MAX_EX_TRAIN         = cfg.get('max_ex_train', None)
    INFERRED_PROPORTION  = cfg.get('inferred_proportion', 0.0)
    FPR                  = cfg.get('fpr', 0.0)
    FNR                  = cfg.get('fnr', 0.0)
    CONSTRAINT_TRUNC     = CONSTRAINT_TYPE.split('-')[0]
    OUTPUT_DIR           = os.path.join(cfg['output_dir'], DATSET_NAME, SENSITIVE_ATTRIBUTE,
                                        CONSTRAINT_TRUNC, BEHAVIOR_POLICY_TYPE, 'cma')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    GLOBAL_RESULTS_PATH  = cfg.get('global_results_path', None)
    if GLOBAL_RESULTS_PATH and INFERRED_PROPORTION and INFERRED_PROPORTION > 0.0:
        base, ext = os.path.splitext(GLOBAL_RESULTS_PATH)
        if not base.endswith("_inferred"):
            GLOBAL_RESULTS_PATH = f"{base}_inferred{ext or '.json'}"
    SEED                 = cfg['seed']

    np.random.seed(SEED)

    assert BEHAVIOR_POLICY_TYPE in ['random', 'tweak1', 'mixed']
    assert CONSTRAINT_TYPE in ['students-ttest', 'welchs-ttest']
    assert DISPARITY_TYPE in ['reward', 'action']

    datasets = joblib.load(BANDIT_DATA_PATH)
    data        = datasets[BEHAVIOR_POLICY_TYPE]
    train_data  = data['train']
    safety_data = data['safety']
    test_data   = data['test']

    input_dim = train_data['context'].shape[1]
    n_actions = max(train_data['actions']['true_label'].max(),
                    train_data['actions']['action'].max()) + 1

    train_X    = train_data['context'].values
    train_a    = train_data['actions']['action'].values
    train_pi_b = train_data['actions']['propensity'].values
    train_r    = train_data['reward'].values
    train_s    = train_data['sensitive'].values

    # Optional: cap training data size before applying train_frac
    if MAX_EX_TRAIN is not None:
        if MAX_EX_TRAIN <= 0:
            raise ValueError(f"max_ex_train must be > 0, got {MAX_EX_TRAIN}")
        n_train = len(train_X)
        if MAX_EX_TRAIN < n_train:
            rng = np.random.RandomState(SEED)
            keep_idx = rng.permutation(n_train)[:MAX_EX_TRAIN]
            train_X = train_X[keep_idx]
            train_a = train_a[keep_idx]
            train_pi_b = train_pi_b[keep_idx]
            train_r = train_r[keep_idx]
            train_s = train_s[keep_idx]
            print(f"Capped train split to {MAX_EX_TRAIN}/{n_train} examples using seed {SEED}")

    # Optional: subsample training data only (safety/test untouched)
    if not (0.0 < TRAIN_FRAC <= 1.0):
        raise ValueError(f"train_frac must be in (0, 1], got {TRAIN_FRAC}")
    if TRAIN_FRAC < 1.0:
        n_train = len(train_X)
        n_keep = max(1, int(n_train * TRAIN_FRAC))
        rng = np.random.RandomState(SEED)
        keep_idx = rng.permutation(n_train)[:n_keep]
        train_X = train_X[keep_idx]
        train_a = train_a[keep_idx]
        train_pi_b = train_pi_b[keep_idx]
        train_r = train_r[keep_idx]
        train_s = train_s[keep_idx]
        print(f"Subsampled train split to {n_keep}/{n_train} ({TRAIN_FRAC:.4f}) using seed {SEED}")

    # Optional: corrupt sensitive labels for inferred-attribute experiments (train only)
    if INFERRED_PROPORTION and INFERRED_PROPORTION > 0.0:
        if not (0.0 < INFERRED_PROPORTION < 1.0):
            raise ValueError(f"inferred_proportion must be in (0, 1), got {INFERRED_PROPORTION}")
        if not (0.0 <= FPR <= 1.0):
            raise ValueError(f"fpr must be in [0, 1], got {FPR}")
        if not (0.0 <= FNR <= 1.0):
            raise ValueError(f"fnr must be in [0, 1], got {FNR}")
        n_train = len(train_s)
        n_inferred = int(n_train * INFERRED_PROPORTION)
        rng = np.random.RandomState(SEED)
        inferred_idx = rng.choice(n_train, size=n_inferred, replace=False)
        print(f"Train split — ground-truth: {n_train - n_inferred}, inferred: {n_inferred} "
              f"(proportion={INFERRED_PROPORTION}, fpr={FPR}, fnr={FNR})")

        sensitive_noisy = train_s.ravel().copy().astype(int)
        inferred_orig   = sensitive_noisy[inferred_idx]

        flip_probs      = np.where(inferred_orig == 0, FPR, FNR)
        flip_mask       = rng.random(n_inferred) < flip_probs
        
        sensitive_noisy[inferred_idx[flip_mask]] = 1 - sensitive_noisy[inferred_idx[flip_mask]]
        n_fp = int(flip_mask[inferred_orig == 0].sum())
        n_fn = int(flip_mask[inferred_orig == 1].sum())
        print(f"Inferred flips — FP (0→1): {n_fp}, FN (1→0): {n_fn}, total: {flip_mask.sum()}")

        train_s = sensitive_noisy
        is_inferred = np.zeros(n_train, dtype=bool)
        is_inferred[inferred_idx] = True
        gt_sensitive     = sensitive_noisy[~is_inferred]
        train_grp_1_size = int(gt_sensitive.sum())
        train_grp_2_size = int((~is_inferred).sum()) - train_grp_1_size
        print(f"Ground-truth partition — grp1 (sens=1): {train_grp_1_size}, grp2 (sens=0): {train_grp_2_size}")

    policy = LinearPolicy(input_dim, n_actions)
    x0     = np.zeros(policy.param_dim)

    safety_data_size        = len(safety_data['context'])
    safety_data_grp_1_size  = safety_data['sensitive'].values.sum().item()
    safety_data_grp_2_size  = safety_data_size - safety_data_grp_1_size

    constraint_kwargs = {
        'fail_prob':               FAIL_PROB,
        'epsilon':                 EPSILON,
        'safety_data_grp_1_size':  safety_data_grp_1_size,
        'safety_data_grp_2_size':  safety_data_grp_2_size,
        'constraint_type':         CONSTRAINT_TYPE,
        'disparity_type':          DISPARITY_TYPE,
    }

    # --- CMA-ES loop placeholder ---
    cma_opts = {'seed': SEED, 'maxiter': MAXITER, 'verbose': 1}
    if POPSIZE is not None:
        cma_opts['popsize'] = POPSIZE

    es = cma.CMAEvolutionStrategy(x0, SIGMA0, cma_opts)
    while not es.stop():
        solutions = es.ask()
        # Objective: negative mean IS return with hard barrier on constraint (CMA-ES minimizes)
        fitnesses = Parallel(n_jobs=N_JOBS)(
            delayed(evaluate_solution)(params, train_X, train_a, train_pi_b, train_r, train_s,
                                       policy, BATCH_SIZE, constraint_kwargs, EPSILON)
            for params in solutions
        )
        es.tell(solutions, fitnesses)
        es.disp()

    best_params = es.result.xbest
    np.save(os.path.join(OUTPUT_DIR, 'best_params.npy'), best_params)
    print(f"Best params norm: {np.linalg.norm(best_params):.4f}")

    # --- Safety Test & Test Evaluation ---
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
        sens   = torch.tensor(s.astype(bool), dtype=torch.bool)
        return is_r, is_a, pi_a_t, sens

    is_r, is_a, pi_a_t, sens = gather_is(safety_data)
    metric = is_r if DISPARITY_TYPE == 'reward' else is_a
    group_1 = metric[sens]
    group_2 = metric[~sens]
    group_1_mean, group_1_std = group_1.mean(), group_1.std(correction=1)
    group_2_mean, group_2_std = group_2.mean(), group_2.std(correction=1)
    if CONSTRAINT_TYPE == 'students-ttest':
        K1 = stats.t.ppf(1 - (FAIL_PROB/4.0), safety_data_grp_1_size - 1) / math.sqrt(safety_data_grp_1_size)
        K2 = stats.t.ppf(1 - (FAIL_PROB/4.0), safety_data_grp_2_size - 1) / math.sqrt(safety_data_grp_2_size)
        ucb_g1_v_g2 = group_1_mean - group_2_mean + K1 * group_1_std + K2 * group_2_std
        ucb_g2_v_g1 = group_2_mean - group_1_mean + K1 * group_1_std + K2 * group_2_std
    elif CONSTRAINT_TYPE == 'welchs-ttest':
        v_num = (group_1_std**2 / safety_data_grp_1_size) + (group_2_std**2 / safety_data_grp_2_size)
        v_den = ((group_1_std**2 / safety_data_grp_1_size)**2 / (safety_data_grp_1_size - 1)) + ((group_2_std**2 / safety_data_grp_2_size)**2 / (safety_data_grp_2_size - 1))
        satterthwaite_v = (v_num / (v_den + 1e-8))
        K = stats.t.ppf(1 - (FAIL_PROB/2.0), satterthwaite_v.item())
        ucb_g1_v_g2 = group_1_mean - group_2_mean + K * torch.sqrt((group_1_std**2 / safety_data_grp_1_size) + (group_2_std**2 / safety_data_grp_2_size))
        ucb_g2_v_g1 = group_2_mean - group_1_mean + K * torch.sqrt((group_1_std**2 / safety_data_grp_1_size) + (group_2_std**2 / safety_data_grp_2_size))
    else:
        raise ValueError(f"Invalid constraint type provided: {CONSTRAINT_TYPE}")
    safety_flag = 'Safe' if (ucb_g1_v_g2.item() < EPSILON and ucb_g2_v_g1.item() < EPSILON) else 'NSF'
    safety_metrics = {
        'ucb_g1_v_g2': ucb_g1_v_g2.item(),
        'ucb_g2_v_g1': ucb_g2_v_g1.item(),
        'epsilon': EPSILON,
        'group_1_mean': group_1_mean.item(),
        'group_2_mean': group_2_mean.item(),
        'group_1_std': group_1_std.item(),
        'group_2_std': group_2_std.item(),
        'n_group_1': int(sens.sum().item()),
        'n_group_2': int((~sens).sum().item()),
    }

    is_r, is_a, pi_a_t, sens = gather_is(test_data)
    metric = is_r if DISPARITY_TYPE == 'reward' else is_a
    group_1 = metric[sens]
    group_2 = metric[~sens]
    test_constraint = torch.abs(group_1.mean() - group_2.mean())
    test_flag = 'Safe' if test_constraint.item() < EPSILON else 'Unsafe'
    test_metrics = {
        'constraint': test_constraint.item(),
        'epsilon': EPSILON,
        'group_1_mean': group_1.mean().item(),
        'group_2_mean': group_2.mean().item(),
        'mean_is_reward': is_r.mean().item(),
        'n_group_1': int(sens.sum().item()),
        'n_group_2': int((~sens).sum().item()),
    }

    verdict = ("Error: Our High Confidence Algorithm failed"
               if safety_flag == 'Safe' and test_flag == 'Unsafe'
               else "Success: Our High Confidence Algorithm succeeded")
    print(verdict)

    results_path = os.path.join(OUTPUT_DIR, 'results.txt')
    with open(results_path, 'w') as f:
        f.write(f"seed:        {SEED}\n")
        if INFERRED_PROPORTION and INFERRED_PROPORTION > 0.0:
            f.write(f"inferred_proportion: {INFERRED_PROPORTION}\n")
            f.write(f"fpr: {FPR}\n")
            f.write(f"fnr: {FNR}\n")
        f.write(f"safety_flag: {safety_flag}\n")
        f.write(f"test_flag:   {test_flag}\n")
        f.write(f"verdict:     {verdict}\n")
        f.write(f"safety_epsilon: {safety_metrics['epsilon']}\n")
        f.write(f"safety_ucb_g1_v_g2: {safety_metrics['ucb_g1_v_g2']}\n")
        f.write(f"safety_ucb_g2_v_g1: {safety_metrics['ucb_g2_v_g1']}\n")
        f.write(f"safety_group_1_mean: {safety_metrics['group_1_mean']}\n")
        f.write(f"safety_group_2_mean: {safety_metrics['group_2_mean']}\n")
        f.write(f"safety_group_1_std: {safety_metrics['group_1_std']}\n")
        f.write(f"safety_group_2_std: {safety_metrics['group_2_std']}\n")
        f.write(f"safety_n_group_1: {safety_metrics['n_group_1']}\n")
        f.write(f"safety_n_group_2: {safety_metrics['n_group_2']}\n")
        f.write(f"test_epsilon: {test_metrics['epsilon']}\n")
        f.write(f"test_constraint: {test_metrics['constraint']}\n")
        f.write(f"test_group_1_mean: {test_metrics['group_1_mean']}\n")
        f.write(f"test_group_2_mean: {test_metrics['group_2_mean']}\n")
        f.write(f"test_n_group_1: {test_metrics['n_group_1']}\n")
        f.write(f"test_n_group_2: {test_metrics['n_group_2']}\n")
        f.write(f"test_mean_is_reward: {test_metrics['mean_is_reward']}\n")
    print(f"Saved results → {results_path}")

    if GLOBAL_RESULTS_PATH:
        os.makedirs(os.path.dirname(GLOBAL_RESULTS_PATH), exist_ok=True)
        record = {
            'seed': SEED,
            'mode': 'cma',
            'train_frac': TRAIN_FRAC,
            'max_ex_train': MAX_EX_TRAIN,
            'inferred_proportion': INFERRED_PROPORTION,
            'fpr': FPR,
            'fnr': FNR,
            'dataset_name': DATSET_NAME,
            'sensitive_attribute': SENSITIVE_ATTRIBUTE,
            'behavior_policy_type': BEHAVIOR_POLICY_TYPE,
            'constraint_type': CONSTRAINT_TYPE,
            'disparity_type': DISPARITY_TYPE,
            'output_dir': OUTPUT_DIR,
            'safety_flag': safety_flag,
            'test_flag': test_flag,
            'verdict': verdict,
            'safety': safety_metrics,
            'test': test_metrics,
        }

        try:
            import fcntl
        except ImportError:
            fcntl = None

        with open(GLOBAL_RESULTS_PATH, 'a+', encoding='utf-8') as f:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            content = f.read().strip()
            data = json.loads(content) if content else {}
            mode_key = 'cma_inferred' if INFERRED_PROPORTION and INFERRED_PROPORTION > 0.0 else 'cma'
            frac_key = str(TRAIN_FRAC)
            seed_key = str(SEED)
            if INFERRED_PROPORTION and INFERRED_PROPORTION > 0.0:
                fpr_key = f"fpr_{FPR}"
                fnr_key = f"fnr_{FNR}"
                data.setdefault(mode_key, {}).setdefault(fpr_key, {}).setdefault(fnr_key, {}).setdefault(frac_key, {})[seed_key] = record
            else:
                data.setdefault(mode_key, {}).setdefault(frac_key, {})[seed_key] = record
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_UN)