import numpy as np
import joblib
from joblib import Parallel, delayed
import os, sys, ipdb
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
    CONSTRAINT_TRUNC     = CONSTRAINT_TYPE.split('-')[0]
    OUTPUT_DIR           = os.path.join(cfg['output_dir'], DATSET_NAME, SENSITIVE_ATTRIBUTE,
                                        CONSTRAINT_TRUNC, BEHAVIOR_POLICY_TYPE, 'cma')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
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
    _, ucb_abs_diff = bound_propagation(is_r, is_a, pi_a_t, sens, constraint_kwargs)
    safety_flag = 'Safe' if ucb_abs_diff.item() < EPSILON else 'NSF'

    is_r, is_a, pi_a_t, sens = gather_is(test_data)
    m = is_r if DISPARITY_TYPE == 'reward' else is_a
    test_flag = 'Safe' if torch.abs(m[sens].mean() - m[~sens].mean()).item() < EPSILON else 'Unsafe'

    verdict = ("Error: Our High Confidence Algorithm failed"
               if safety_flag == 'Safe' and test_flag == 'Unsafe'
               else "Success: Our High Confidence Algorithm succeeded")
    print(verdict)

    results_path = os.path.join(OUTPUT_DIR, 'results.txt')
    with open(results_path, 'w') as f:
        f.write(f"seed:        {SEED}\n")
        f.write(f"safety_flag: {safety_flag}\n")
        f.write(f"test_flag:   {test_flag}\n")
        f.write(f"verdict:     {verdict}\n")
    print(f"Saved results → {results_path}")
