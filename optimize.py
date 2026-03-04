import pandas as pd
import numpy as np
import torch
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from policy import PolicyNet
from utils import students_ttest_ucb_constraint, welchs_ttest_ucb_constraint, safety_test, evaluate_on_test
import math
import joblib
import os, sys, ipdb
import argparse
import yaml
import wandb
from tqdm import tqdm


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--mode',                type=str,   default=None, choices=['seldonian', 'naive', 'fixed_lambda', 'lagrange_exp'])
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
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI args override YAML values when explicitly provided
    for key, val in vars(args).items():
        if key != 'config' and val is not None:
            cfg[key] = val

    MODE                 = cfg['mode']
    assert MODE in ['seldonian', 'naive', 'fixed_lambda', 'lagrange_exp'], f"Invalid mode: {MODE}"
    FIXED_LAMBDA         = cfg.get('fixed_lambda', 1.0)
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
    CONSTRAINT_TRUNC     = CONSTRAINT_TYPE.split('-')[0]
    OUTPUT_DIR           = os.path.join(cfg['output_dir'], DATSET_NAME, SENSITIVE_ATTRIBUTE,
                                        CONSTRAINT_TRUNC, BEHAVIOR_POLICY_TYPE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    SEED                 = cfg['seed']

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert CONSTRAINT_TYPE in ['students-ttest', 'welchs-ttest'], f"Invalid constraint type provided: {CONSTRAINT_TYPE}"
    assert DISPARITY_TYPE in ['reward', 'action'], f"Invalid disparity type provided: {DISPARITY_TYPE}"
    
    wandb.init(
        project="group-fair-bandits",
        name=f"{DATSET_NAME}-{SENSITIVE_ATTRIBUTE}-{MODE}-{CONSTRAINT_TRUNC}-{BEHAVIOR_POLICY_TYPE}",
        config={
            "dataset_name": DATSET_NAME,
            "run_mode": MODE,
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
            "dual_opt_lr": DUAL_OPT_LR
        },
    )

    assert BEHAVIOR_POLICY_TYPE in ['random', 'tweak1', 'mixed'], f"Invalid behavior policy type provided: {BEHAVIOR_POLICY_TYPE}"

    datasets = joblib.load(BANDIT_DATA_PATH)
    # datasets: {policy_name: {'train': bandit_dict, 'safety': bandit_dict, 'test': bandit_dict}}
    data = datasets[BEHAVIOR_POLICY_TYPE]
    train_data  = data['train']
    safety_data = data['safety']
    test_data   = data['test']

    # Setup Hyperparameters
    input_dim = train_data['context'].shape[1]  # Number of context features
    n_actions = max(train_data['actions']['true_label'].max(),
                train_data['actions']['action'].max()) + 1
    policy = PolicyNet(input_dim, n_actions).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=PARAM_OPT_LR)

    # Lagrangian multipliers in log space (exp ensures positivity) — seldonian and lagrange_exp
    if MODE in ('seldonian', 'lagrange_exp'):
        log_lambda1 = torch.zeros(1, requires_grad=True, device=device)
        log_lambda2 = torch.zeros(1, requires_grad=True, device=device)
        lambda_optimizer = torch.optim.Adam([log_lambda1, log_lambda2], lr=DUAL_OPT_LR)

    safety_data_size = len(safety_data['context'])
    safety_data_grp_1_size = safety_data['sensitive'].values.sum().item()
    safety_data_grp_2_size = safety_data_size - safety_data_grp_1_size

    epoch_iter = tqdm(range(NUM_EPOCHS), desc="Training")
    for epoch in epoch_iter:
        # 1. Generate shuffled indices for this epoch
        indices = np.arange(len(train_data['context']))
        np.random.shuffle(indices)

        # 2. Reorder data using the shuffled indices
        # (Assuming these are pandas DataFrames/Series or NumPy arrays)
        contexts_shuffled  = train_data['context'].iloc[indices]
        actions_shuffled   = train_data['actions'][['action', 'propensity']].iloc[indices]
        true_lables_shuffled = train_data['actions']['true_label'].iloc[indices]
        rewards_shuffled   = train_data['reward'].iloc[indices]
        sensitive_shuffled = train_data['sensitive'].iloc[indices]
        
        # 3. Iterate over the data in batches
        batch_iter = tqdm(enumerate(range(0, len(contexts_shuffled), BATCH_SIZE)),
                          total=len(contexts_shuffled) // BATCH_SIZE,
                          desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for i, b_sidx in batch_iter:
            
            step_idx = (len(contexts_shuffled)//BATCH_SIZE)*epoch + i
            # Slice the batch
            batch_X = contexts_shuffled.iloc[b_sidx : b_sidx + BATCH_SIZE]
            batch_a = actions_shuffled.iloc[b_sidx : b_sidx + BATCH_SIZE]
            batch_r = rewards_shuffled.iloc[b_sidx : b_sidx + BATCH_SIZE]
            batch_s = sensitive_shuffled.iloc[b_sidx : b_sidx + BATCH_SIZE]
            batch_true_labels = true_lables_shuffled.iloc[b_sidx : b_sidx + BATCH_SIZE]
            
            # --- Your Training Logic Here ---
            # Mask, dim => (B,)
            sensitive_mask = torch.tensor(batch_s.values, dtype=torch.long).to(device)

            if MODE in ('seldonian', 'fixed_lambda', 'lagrange_exp') and (sensitive_mask.sum().item() == 0 or sensitive_mask.sum().item() == BATCH_SIZE):
                print(f"Warning: One of the groups has no samples in this batch. Skipping fairness constraint for this batch.")
                continue
            
            # X => (B, D), action => (B,1), reward => (B,1), propensity => (B,1)
            batch_X_tensor = torch.tensor(batch_X.values, dtype=torch.float32).to(device)  # Add a dimension for batch processing
            pi_b_actions = torch.tensor(batch_a['action'].values, dtype=torch.long).unsqueeze(1).to(device)
            pi_b_action_probs = torch.tensor(batch_a['propensity'].values, dtype=torch.float32).unsqueeze(1).to(device)
            pi_b_rewards = torch.tensor(batch_r.values, dtype=torch.float32).unsqueeze(1).to(device)
            true_batch_labels = torch.tensor(batch_true_labels.values, dtype=torch.long).unsqueeze(1).to(device)
            
            output_probs = policy(batch_X_tensor)
            # output_probs_a => (B,1) - the probability of the action that was actually taken by our policy
            output_probs_a = output_probs.gather(1, pi_b_actions)

            # Objective Term
            is_reward = (output_probs_a.detach() / pi_b_action_probs) * pi_b_rewards
            reward_obj_term = torch.mean(is_reward * torch.log(output_probs_a + 1e-8))  # Add small epsilon for stability
            # CHECK IF THIS IS CORRECT FOR ACTION DISPARITY!!
            is_action = (output_probs_a.detach() / pi_b_action_probs) * (pi_b_actions.float())

            constraint_kwargs = {
                'fail_prob': FAIL_PROB,
                'epsilon': EPSILON,
                'fixed_lambda': FIXED_LAMBDA,
                'safety_data_grp_1_size': safety_data_grp_1_size,
                'safety_data_grp_2_size': safety_data_grp_2_size,
                'constraint_type': CONSTRAINT_TYPE,
                'disparity_type': DISPARITY_TYPE
            }

            if MODE == 'seldonian':
                if CONSTRAINT_TYPE == 'students-ttest':
                    constraint_term, metrics = students_ttest_ucb_constraint(is_reward, is_action, output_probs_a, sensitive_mask, log_lambda1, log_lambda2, constraint_kwargs)
                elif CONSTRAINT_TYPE == 'welchs-ttest':
                    constraint_term, metrics = welchs_ttest_ucb_constraint(is_reward, is_action, output_probs_a, sensitive_mask, log_lambda1, log_lambda2, constraint_kwargs)
                else:
                    raise ValueError(f"Invalid constraint type provided: {CONSTRAINT_TYPE}")

                lambda_optimizer.zero_grad()
                lambda_loss = -((metrics['ucb_g1_v_g2'] - EPSILON) * log_lambda1.exp()) - ((metrics['ucb_g2_v_g1'] - EPSILON) * log_lambda2.exp())
                lambda_loss.backward()
                lambda_optimizer.step()
                with torch.no_grad():
                    log_lambda1.clamp_(max=LOG_LAMBDA_MAX)
                    log_lambda2.clamp_(max=LOG_LAMBDA_MAX)

            elif MODE == 'fixed_lambda':
                constraint_term, metrics = students_ttest_ucb_constraint(is_reward, is_action, output_probs_a, sensitive_mask, None, None, constraint_kwargs, no_ucb=True)
                
            elif MODE == 'lagrange_exp':
                constraint_term, metrics = students_ttest_ucb_constraint(is_reward, is_action, output_probs_a, sensitive_mask, log_lambda1, log_lambda2, constraint_kwargs, lagrange_exp=True)
                lambda_optimizer.zero_grad()
                lambda_loss = -((metrics['raw_g1_v_g2'] - EPSILON) * log_lambda1.exp()) - ((metrics['raw_g2_v_g1'] - EPSILON) * log_lambda2.exp())
                lambda_loss.backward()
                lambda_optimizer.step()
                with torch.no_grad():
                    log_lambda1.clamp_(max=LOG_LAMBDA_MAX)
                    log_lambda2.clamp_(max=LOG_LAMBDA_MAX)

            optimizer.zero_grad()
            if MODE in ('seldonian', 'fixed_lambda', 'lagrange_exp'):
                loss = -(reward_obj_term - constraint_term)
            else:  # naive
                loss = -reward_obj_term
            loss.backward()
            optimizer.step()

            log_dict = {"train/loss": loss.detach().item(), "train/is_reward_mean": is_reward.mean().item()}
            if MODE == 'seldonian':
                log_dict.update({
                    "train/lambda1": log_lambda1.detach().exp().item(),
                    "train/lambda2": log_lambda2.detach().exp().item(),
                    "train/ucb_g1_v_g2": metrics['ucb_g1_v_g2'].item(),
                    "train/ucb_g2_v_g1": metrics['ucb_g2_v_g1'].item(),
                    "train/mean_g1": metrics['group_1_is_metric_mean'].item(),
                    "train/mean_g2": metrics['group_2_is_metric_mean'].item(),
                    "train/stdev_g1": metrics['group_1_is_metric_std'].item(),
                    "train/stdev_g2": metrics['group_2_is_metric_std'].item(),
                })
            elif MODE == 'fixed_lambda':
                log_dict.update({
                    "train/mean_g1": metrics['group_1_is_metric_mean'].item(),
                    "train/mean_g2": metrics['group_2_is_metric_mean'].item(),
                })
            elif MODE == 'lagrange_exp':
                log_dict.update({
                    "train/lambda1": log_lambda1.detach().exp().item(),
                    "train/lambda2": log_lambda2.detach().exp().item(),
                    "train/mean_g1": metrics['group_1_is_metric_mean'].item(),
                    "train/mean_g2": metrics['group_2_is_metric_mean'].item(),
                })

            if(step_idx % EVAL_FREQ == 0):
                with torch.no_grad():
                    output_probs_eval = policy(batch_X_tensor)
                    predicted_actions_eval = torch.argmax(output_probs_eval, dim=1)  # Get the action with highest probability for each context
                    match_eval = predicted_actions_eval.squeeze() == true_batch_labels.squeeze()
                    rewards_eval = torch.where(match_eval, torch.tensor(1.0), torch.tensor(-1.0))
                    mean_reward_eval = rewards_eval.mean()

                log_dict["eval/mean_reward_eval_on_policy"] = mean_reward_eval.item()

                if MODE in ('seldonian', 'fixed_lambda', 'lagrange_exp'):
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

    # Save trained policy
    torch.save(policy.state_dict(), os.path.join(OUTPUT_DIR, 'policy.pt'))
    print(f"Saved policy → {os.path.join(OUTPUT_DIR, 'policy.pt')}")

    # Perform Safety Test
    safety_flag = safety_test(policy, safety_data, device, constraint_kwargs, batch_size=BATCH_SIZE)

    # Verify test data
    test_flag = evaluate_on_test(policy, test_data, device, constraint_kwargs, batch_size=BATCH_SIZE)

    if safety_flag == 'Safe' and test_flag == 'Unsafe':
        verdict = "Error: Our High Confidence Algorithm failed"
    else:
        verdict = "Success: Our High Confidence Algorithm succeeded"
    print(verdict)

    results_path = os.path.join(OUTPUT_DIR, 'results.txt')
    with open(results_path, 'w') as f:
        f.write(f"seed:        {SEED}\n")
        f.write(f"safety_flag: {safety_flag}\n")
        f.write(f"test_flag:   {test_flag}\n")
        f.write(f"verdict:     {verdict}\n")
    print(f"Saved results → {results_path}")