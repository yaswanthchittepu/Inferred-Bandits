import pandas as pd
import numpy as np
import torch
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from policy import PolicyNet
from utils import safety_test, evaluate_on_test, bound_prop_ucb_constraint_no_safety
import math
import joblib
import os, sys, ipdb
import json
import argparse
import yaml
import wandb
from tqdm import tqdm

# Inferred-attributes Seldonian: train with mixed ground-truth + inferred sensitive labels.
# Safety data is assumed to have no inferred attributes (ground truth only).

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default_inferred_no_safe.yaml')
    parser.add_argument('--mode',                type=str,   default=None, choices=['seldonian', 'naive'])
    parser.add_argument('--fixed_lambda',        type=float, default=None)
    parser.add_argument('--seed',                type=int,   default=None)
    parser.add_argument('--output_dir',          type=str,   default=None)
    parser.add_argument('--dataset_name',        type=str,   default=None)
    parser.add_argument('--sensitive_attribute',  type=str,   default=None)
    parser.add_argument('--behavior_policy_type', type=str,   default=None)
    parser.add_argument('--bandit_data_dir',      type=str,   default=None)
    parser.add_argument('--constraint_type',      type=str,   default=None)
    parser.add_argument('--disparity_type',       type=str,   default=None)
    parser.add_argument('--num_epochs',           type=int,   default=None)
    parser.add_argument('--batch_size',           type=int,   default=None)
    parser.add_argument('--param_opt_lr',         type=float, default=None)
    parser.add_argument('--dual_opt_lr',          type=float, default=None)
    parser.add_argument('--epsilon',              type=float, default=None)
    parser.add_argument('--fail_prob',            type=float, default=None)
    parser.add_argument('--log_lambda_max',       type=float, default=None)
    parser.add_argument('--eval_freq',            type=int,   default=None)
    parser.add_argument('--inferred_proportion',   type=float, default=None)
    parser.add_argument('--fpr',                   type=float, default=None)
    parser.add_argument('--fnr',                   type=float, default=None)
    parser.add_argument('--hidden_dim',            type=int,   default=None)
    parser.add_argument('--global_results_path',   type=str,   default=None)
    parser.add_argument('--train_frac',            type=float, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI args override YAML values when explicitly provided
    for key, val in vars(args).items():
        if key != 'config' and val is not None:
            cfg[key] = val

    MODE                 = cfg['mode']
    assert MODE in ['seldonian', 'naive'], f"Invalid mode: {MODE}"
    DATSET_NAME          = cfg['dataset_name']
    SENSITIVE_ATTRIBUTE  = cfg['sensitive_attribute']
    BEHAVIOR_POLICY_TYPE = cfg['behavior_policy_type']
    BANDIT_DATA_PATH     = os.path.join(cfg['bandit_data_dir'], DATSET_NAME, SENSITIVE_ATTRIBUTE, 'datasets.joblib')
    CONSTRAINT_TYPE      = cfg['constraint_type']
    DISPARITY_TYPE       = cfg['disparity_type']
    NUM_EPOCHS           = cfg['num_epochs']
    BATCH_SIZE           = cfg['batch_size']
    PARAM_OPT_LR         = cfg['param_opt_lr']
    DUAL_OPT_LR          = cfg['dual_opt_lr']
    EPSILON              = cfg['epsilon']
    FAIL_PROB            = cfg['fail_prob']
    LOG_LAMBDA_MAX       = cfg['log_lambda_max']
    EVAL_FREQ            = cfg['eval_freq']
    INFERRED_PROPORTION  = cfg['inferred_proportion']
    FPR                  = cfg['fpr']   # alpha: P(inferred=1 | true=0)
    FNR                  = cfg['fnr']   # beta:  P(inferred=0 | true=1)
    HIDDEN_DIM           = cfg.get('hidden_dim', 128)
    TRAIN_FRAC           = cfg.get('train_frac', 1.0)
    CONSTRAINT_TRUNC     = CONSTRAINT_TYPE.split('-')[0]
    OUTPUT_DIR           = os.path.join(cfg['output_dir'], DATSET_NAME, SENSITIVE_ATTRIBUTE,
                                        CONSTRAINT_TRUNC, BEHAVIOR_POLICY_TYPE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    GLOBAL_RESULTS_PATH  = cfg.get('global_results_path', None)
    if GLOBAL_RESULTS_PATH and INFERRED_PROPORTION and INFERRED_PROPORTION > 0.0:
        base, ext = os.path.splitext(GLOBAL_RESULTS_PATH)
        if not base.endswith("_inferred"):
            GLOBAL_RESULTS_PATH = f"{base}_inferred{ext or '.json'}"
    SEED                 = cfg['seed']

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert CONSTRAINT_TYPE in ['students-ttest', 'welchs-ttest'], f"Invalid constraint type provided: {CONSTRAINT_TYPE}"
    assert DISPARITY_TYPE in ['reward', 'action'], f"Invalid disparity type provided: {DISPARITY_TYPE}"

    wandb.init(
        project="group-fair-bandits",
        name=f"{DATSET_NAME}-{SENSITIVE_ATTRIBUTE}-inferred-{MODE}-{CONSTRAINT_TRUNC}-{BEHAVIOR_POLICY_TYPE}",
        config={
            "dataset_name": DATSET_NAME,
            "run_mode": f"inferred_{MODE}",
            "sensitive_attribute": SENSITIVE_ATTRIBUTE,
            "behavior_policy_type": BEHAVIOR_POLICY_TYPE,
            "constraint_type": CONSTRAINT_TYPE,
            "disparity_type": DISPARITY_TYPE,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "epsilon": EPSILON,
            "fail_prob": FAIL_PROB,
            "log_lambda_max": LOG_LAMBDA_MAX,
            "eval_freq": EVAL_FREQ,
            "param_opt_lr": PARAM_OPT_LR,
            "dual_opt_lr": DUAL_OPT_LR,
            "inferred_proportion": INFERRED_PROPORTION,
            "FPR": FPR,
            "FNR": FNR,
            "hidden_dim": HIDDEN_DIM,
        },
    )

    assert BEHAVIOR_POLICY_TYPE in ['random', 'tweak1', 'mixed'], f"Invalid behavior policy type provided: {BEHAVIOR_POLICY_TYPE}"

    datasets = joblib.load(BANDIT_DATA_PATH)
    data = datasets[BEHAVIOR_POLICY_TYPE]
    train_data  = data['train']
    safety_data = data['safety']
    test_data   = data['test']

    # Optional: subsample training data (same as optimize.py / optimize_cma.py)
    if not (0.0 < TRAIN_FRAC <= 1.0):
        raise ValueError(f"train_frac must be in (0, 1], got {TRAIN_FRAC}")
    if TRAIN_FRAC < 1.0:
        n_full = len(train_data['context'])
        n_keep = max(1, int(n_full * TRAIN_FRAC))
        rng = np.random.RandomState(SEED)
        keep_idx = rng.permutation(n_full)[:n_keep]
        train_data = {
            'context': train_data['context'].iloc[keep_idx].reset_index(drop=True),
            'actions': train_data['actions'].iloc[keep_idx].reset_index(drop=True),
            'reward': train_data['reward'].iloc[keep_idx].reset_index(drop=True),
            'sensitive': train_data['sensitive'].iloc[keep_idx].reset_index(drop=True),
        }
        print(f"Subsampled train to {n_keep}/{n_full} (train_frac={TRAIN_FRAC})")

    # Split train_data into ground-truth and inferred partitions.
    # The inferred partition has its sensitive attribute corrupted at FPR/FNR.
    n_train      = len(train_data['context'])
    n_inferred   = int(n_train * INFERRED_PROPORTION)
    inferred_idx = np.random.choice(n_train, size=n_inferred, replace=False)
    print(f"Train split — ground-truth: {n_train - n_inferred}, inferred: {n_inferred} "
          f"(proportion={INFERRED_PROPORTION}, fpr={FPR}, fnr={FNR})")

    sensitive_noisy = train_data['sensitive'].values.ravel().copy().astype(int)
    inferred_orig   = sensitive_noisy[inferred_idx]
    flip_probs      = np.where(inferred_orig == 0, FPR, FNR)
    flip_mask       = np.random.random(n_inferred) < flip_probs
    sensitive_noisy[inferred_idx[flip_mask]] = 1 - sensitive_noisy[inferred_idx[flip_mask]]
    n_fp = int(flip_mask[inferred_orig == 0].sum())
    n_fn = int(flip_mask[inferred_orig == 1].sum())
    print(f"Inferred flips — FP (0→1): {n_fp}, FN (1→0): {n_fn}, total: {flip_mask.sum()}")
    train_data['sensitive'] = pd.Series(
        sensitive_noisy, index=train_data['sensitive'].index,
    )
    is_inferred = np.zeros(n_train, dtype=bool)
    is_inferred[inferred_idx] = True
    train_data['is_inferred'] = pd.Series(is_inferred, index=train_data['sensitive'].index)

    # Group sizes on ground-truth partition only (sensitive=1 → grp1, sensitive=0 → grp2)
    gt_sensitive     = sensitive_noisy[~is_inferred]
    train_grp_1_size = int(gt_sensitive.sum())
    train_grp_2_size = int((~is_inferred).sum()) - train_grp_1_size
    print(f"Ground-truth partition — grp1 (sens=1): {train_grp_1_size}, grp2 (sens=0): {train_grp_2_size}")

    input_dim = train_data['context'].shape[1]
    n_actions = max(train_data['actions']['true_label'].max(),
                    train_data['actions']['action'].max()) + 1
    policy = PolicyNet(input_dim, n_actions, hidden_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=PARAM_OPT_LR)

    if MODE == 'seldonian':
        log_lambda1 = torch.zeros(1, requires_grad=True, device=device)
        log_lambda2 = torch.zeros(1, requires_grad=True, device=device)
        lambda_optimizer = torch.optim.Adam([log_lambda1, log_lambda2], lr=DUAL_OPT_LR)

    safety_data_size = len(safety_data['context'])
    safety_data_grp_1_size = safety_data['sensitive'].values.sum().item()
    safety_data_grp_2_size = safety_data_size - safety_data_grp_1_size

    constraint_kwargs_base = {
        'fail_prob': FAIL_PROB,
        'epsilon': EPSILON,
        'safety_data_grp_1_size': safety_data_grp_1_size,
        'safety_data_grp_2_size': safety_data_grp_2_size,
        'constraint_type': CONSTRAINT_TYPE,
        'disparity_type': DISPARITY_TYPE,
        'inferred_train_data_size':      n_inferred,
        'ground_truth_train_data_size':  n_train - n_inferred,
        'grp1_cnt_train_data': train_grp_1_size,
        'grp2_cnt_train_data': train_grp_2_size,
        'fpr': FPR,
        'fnr': FNR,
    }

    epoch_iter = tqdm(range(NUM_EPOCHS), desc="Training")
    for epoch in epoch_iter:
        indices = np.arange(len(train_data['context']))
        np.random.shuffle(indices)

        contexts_shuffled    = train_data['context'].iloc[indices]
        actions_shuffled     = train_data['actions'][['action', 'propensity']].iloc[indices]
        true_labels_shuffled = train_data['actions']['true_label'].iloc[indices]
        rewards_shuffled     = train_data['reward'].iloc[indices]
        sensitive_shuffled   = train_data['sensitive'].iloc[indices]
        is_inferred_shuffled = train_data['is_inferred'].iloc[indices]

        batch_iter = tqdm(enumerate(range(0, len(contexts_shuffled), BATCH_SIZE)),
                          total=len(contexts_shuffled) // BATCH_SIZE,
                          desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for i, b_sidx in batch_iter:

            step_idx = (len(contexts_shuffled) // BATCH_SIZE) * epoch + i
            batch_X = contexts_shuffled.iloc[b_sidx : b_sidx + BATCH_SIZE]
            batch_a = actions_shuffled.iloc[b_sidx : b_sidx + BATCH_SIZE]
            batch_r = rewards_shuffled.iloc[b_sidx : b_sidx + BATCH_SIZE]
            batch_s = sensitive_shuffled.iloc[b_sidx : b_sidx + BATCH_SIZE]
            batch_true_labels  = true_labels_shuffled.iloc[b_sidx : b_sidx + BATCH_SIZE]
            batch_is_inferred  = is_inferred_shuffled.iloc[b_sidx : b_sidx + BATCH_SIZE]

            sensitive_mask = torch.tensor(batch_s.values, dtype=torch.long).to(device)
            inferred_mask  = torch.tensor(batch_is_inferred.values, dtype=torch.bool).to(device)

            if MODE == 'seldonian' and (sensitive_mask.sum().item() == 0 or sensitive_mask.sum().item() == BATCH_SIZE):
                print("Warning: One of the sensitive groups has no samples in this batch. Skipping fairness constraint for this batch.")
                continue
            if MODE == 'seldonian' and (inferred_mask.sum().item() == 0 or inferred_mask.sum().item() == BATCH_SIZE):
                print("Warning: One of the inferred groups has no samples in this batch. Skipping fairness constraint for this batch.")
                continue

            batch_X_tensor = torch.tensor(batch_X.values, dtype=torch.float32).to(device)
            pi_b_actions = torch.tensor(batch_a['action'].values, dtype=torch.long).unsqueeze(1).to(device)
            pi_b_action_probs = torch.tensor(batch_a['propensity'].values, dtype=torch.float32).unsqueeze(1).to(device)
            pi_b_rewards = torch.tensor(batch_r.values, dtype=torch.float32).unsqueeze(1).to(device)
            true_batch_labels = torch.tensor(batch_true_labels.values, dtype=torch.long).unsqueeze(1).to(device)

            output_probs = policy(batch_X_tensor)
            output_probs_a = output_probs.gather(1, pi_b_actions)

            is_reward = (output_probs_a.detach() / pi_b_action_probs) * pi_b_rewards
            reward_obj_term = torch.mean(is_reward * torch.log(output_probs_a + 1e-8))
            is_action = (output_probs_a.detach() / pi_b_action_probs) * (pi_b_actions.float())

            constraint_kwargs = constraint_kwargs_base.copy()

            if MODE == 'seldonian':
                constraint_term, metrics = bound_prop_ucb_constraint_no_safety(
                    is_reward, is_action, output_probs_a, sensitive_mask, inferred_mask,
                    log_lambda1, log_lambda2, constraint_kwargs)
                lambda_optimizer.zero_grad()
                lambda_loss = -((metrics['ucb_g1_v_g2'] - EPSILON) * log_lambda1.exp()) \
                            - ((metrics['ucb_g2_v_g1'] - EPSILON) * log_lambda2.exp())
                lambda_loss.backward()
                lambda_optimizer.step()
                with torch.no_grad():
                    log_lambda1.clamp_(max=LOG_LAMBDA_MAX)
                    log_lambda2.clamp_(max=LOG_LAMBDA_MAX)

            optimizer.zero_grad()
            if MODE == 'seldonian':
                loss = -(reward_obj_term - constraint_term)
            else:
                loss = -reward_obj_term
            loss.backward()
            optimizer.step()

            log_dict = {"train/loss": loss.detach().item(), "train/is_reward_mean": is_reward.mean().item()}
            if MODE == 'seldonian':
                log_dict.update({
                    "train/lambda1":     log_lambda1.detach().exp().item(),
                    "train/lambda2":     log_lambda2.detach().exp().item(),
                    "train/ucb_g1_v_g2": metrics['ucb_g1_v_g2'].item(),
                    "train/ucb_g2_v_g1": metrics['ucb_g2_v_g1'].item(),
                    "train/mean_g1":     metrics['group_1_is_metric_mean'].item(),
                    "train/mean_g2":     metrics['group_2_is_metric_mean'].item(),
                })

            if step_idx % EVAL_FREQ == 0:
                with torch.no_grad():
                    output_probs_eval = policy(batch_X_tensor)
                    predicted_actions_eval = torch.argmax(output_probs_eval, dim=1)
                    match_eval = predicted_actions_eval.squeeze() == true_batch_labels.squeeze()
                    rewards_eval = torch.where(match_eval, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))
                    mean_reward_eval = rewards_eval.mean()
                log_dict["eval/mean_reward_eval_on_policy"] = mean_reward_eval.item()
                if MODE == 'seldonian':
                    grp_1_rewards_eval = rewards_eval[sensitive_mask]
                    grp_2_rewards_eval = rewards_eval[~sensitive_mask]
                    grp_1_actions_eval = predicted_actions_eval[sensitive_mask]
                    grp_2_actions_eval = predicted_actions_eval[~sensitive_mask]
                    if DISPARITY_TYPE == 'reward':
                        constraint_eval = torch.abs(grp_1_rewards_eval.mean() - grp_2_rewards_eval.mean()) - EPSILON
                    elif DISPARITY_TYPE == 'action':
                        constraint_eval = torch.abs(grp_1_actions_eval.float().mean() - grp_2_actions_eval.float().mean()) - EPSILON
                    else:
                        raise ValueError(f"Invalid disparity type provided: {DISPARITY_TYPE}")
                    log_dict["eval/constraint_eval_on_policy"] = constraint_eval.item()

            wandb.log(log_dict, step=step_idx)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} complete.")

    torch.save(policy.state_dict(), os.path.join(OUTPUT_DIR, 'policy.pt'))
    print(f"Saved policy → {os.path.join(OUTPUT_DIR, 'policy.pt')}")

    safety_flag, safety_metrics = safety_test(
        policy, safety_data, device, constraint_kwargs_base, batch_size=BATCH_SIZE, return_metrics=True
    )
    test_flag, test_metrics = evaluate_on_test(
        policy, test_data, device, constraint_kwargs_base, batch_size=BATCH_SIZE, return_metrics=True
    )

    if safety_flag == 'Safe' and test_flag == 'Unsafe':
        verdict = "Error: Our High Confidence Algorithm failed"
    else:
        verdict = "Success: Our High Confidence Algorithm succeeded"
    print(verdict)

    results_path = os.path.join(OUTPUT_DIR, 'results.txt')
    with open(results_path, 'w') as f:
        f.write(f"seed:        {SEED}\n")
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
            'mode': f'inferred_{MODE}',
            'train_frac': TRAIN_FRAC,
            'dataset_name': DATSET_NAME,
            'sensitive_attribute': SENSITIVE_ATTRIBUTE,
            'behavior_policy_type': BEHAVIOR_POLICY_TYPE,
            'constraint_type': CONSTRAINT_TYPE,
            'disparity_type': DISPARITY_TYPE,
            'output_dir': OUTPUT_DIR,
            'inferred_proportion': INFERRED_PROPORTION,
            'fpr': FPR,
            'fnr': FNR,
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
            mode_key = f"inferred_{MODE}"
            frac_key = str(TRAIN_FRAC)
            seed_key = str(SEED)
            fpr_key = f"fpr_{FPR}"
            fnr_key = f"fnr_{FNR}"
            data.setdefault(mode_key, {}).setdefault(fpr_key, {}).setdefault(fnr_key, {}).setdefault(frac_key, {})[seed_key] = record
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_UN)
