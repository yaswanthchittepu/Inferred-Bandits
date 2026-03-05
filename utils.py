import torch
import torch.nn as nn
import torch.nn.functional as f
from scipy import stats
import math

def students_ttest_ucb_constraint(is_reward, is_action, output_probs_a, sensitive_mask, log_lambda1, log_lambda2, constraint_kwargs, no_ucb=False, lagrange_exp=False):

    # Constraint Terms
    if(constraint_kwargs['disparity_type'] == 'reward'):
        group_1_is_metric = is_reward[sensitive_mask]
        group_2_is_metric = is_reward[~sensitive_mask]
    elif(constraint_kwargs['disparity_type'] == 'action'):
        group_1_is_metric = is_action[sensitive_mask]
        group_2_is_metric = is_action[~sensitive_mask]
    else:
        raise ValueError(f"Invalid disparity type provided: {constraint_kwargs['disparity_type']}")

    group_1_is_metric_mean, group_1_is_metric_std = group_1_is_metric.mean(), group_1_is_metric.std(correction=1)
    group_2_is_metric_mean, group_2_is_metric_std = group_2_is_metric.mean(), group_2_is_metric.std(correction=1)
    group_1_probs_a = output_probs_a[sensitive_mask]
    group_2_probs_a = output_probs_a[~sensitive_mask]

    # Expectation Term in constraint
    group_1_expectation_term = torch.mean(group_1_is_metric * torch.log(group_1_probs_a + 1e-8))
    group_2_expectation_term = torch.mean(group_2_is_metric * torch.log(group_2_probs_a + 1e-8))

    if no_ucb:
        fixed_lambda = constraint_kwargs['fixed_lambda']
        sign = torch.sign(group_1_is_metric_mean - group_2_is_metric_mean).detach()
        constraint_term = fixed_lambda * sign * (group_1_expectation_term - group_2_expectation_term)
        metrics = {
            'group_1_is_metric_mean': group_1_is_metric_mean.detach(),
            'group_1_is_metric_std': group_1_is_metric_std.detach(),
            'group_2_is_metric_mean': group_2_is_metric_mean.detach(),
            'group_2_is_metric_std': group_2_is_metric_std.detach(),
        }
        return constraint_term, metrics

    expectation_obs_term = (torch.exp(log_lambda1).detach().item()-torch.exp(log_lambda2).detach().item()) * (group_1_expectation_term - group_2_expectation_term)

    if lagrange_exp:
        constraint_term = expectation_obs_term
        with torch.no_grad():
            raw_g1_v_g2 = group_1_is_metric_mean - group_2_is_metric_mean
            raw_g2_v_g1 = group_2_is_metric_mean - group_1_is_metric_mean
        metrics = {
            'raw_g1_v_g2': raw_g1_v_g2.detach(),
            'raw_g2_v_g1': raw_g2_v_g1.detach(),
            'group_1_is_metric_mean': group_1_is_metric_mean.detach(),
            'group_1_is_metric_std': group_1_is_metric_std.detach(),
            'group_2_is_metric_mean': group_2_is_metric_mean.detach(),
            'group_2_is_metric_std': group_2_is_metric_std.detach(),
        }
        return constraint_term, metrics

    # Stdev Terms in constraint
    group_1_norm_is_metric = (group_1_is_metric - group_1_is_metric_mean) / (group_1_is_metric_std + 1e-8)
    group_2_norm_is_metric = (group_2_is_metric - group_2_is_metric_mean) / (group_2_is_metric_std + 1e-8)
    group_1_stdev_term = torch.mean(group_1_norm_is_metric * group_1_is_metric * torch.log(group_1_probs_a + 1e-8))
    group_2_stdev_term = torch.mean(group_2_norm_is_metric * group_2_is_metric * torch.log(group_2_probs_a + 1e-8))
    K1 = stats.t.ppf(1 - (constraint_kwargs['fail_prob']/4.0), constraint_kwargs['safety_data_grp_1_size']-1)/math.sqrt(constraint_kwargs['safety_data_grp_1_size'])
    K2 = stats.t.ppf(1 - (constraint_kwargs['fail_prob']/4.0), constraint_kwargs['safety_data_grp_2_size']-1)/math.sqrt(constraint_kwargs['safety_data_grp_2_size'])
    stdev_obs_term = (torch.exp(log_lambda1).detach().item()+torch.exp(log_lambda2).detach().item()) * ((K1*group_1_stdev_term) + (K2*group_2_stdev_term))

    constraint_term = expectation_obs_term + stdev_obs_term

    with torch.no_grad():
        ucb_g1_v_g2 = group_1_is_metric_mean - group_2_is_metric_mean + K1*group_1_is_metric_std + K2*group_2_is_metric_std
        ucb_g2_v_g1 = group_2_is_metric_mean - group_1_is_metric_mean + K1*group_1_is_metric_std + K2*group_2_is_metric_std

    metrics = {
        'ucb_g1_v_g2': ucb_g1_v_g2.detach(),
        'ucb_g2_v_g1': ucb_g2_v_g1.detach(),
        'group_1_is_metric_mean': group_1_is_metric_mean.detach(),
        'group_1_is_metric_std': group_1_is_metric_std.detach(),
        'group_2_is_metric_mean': group_2_is_metric_mean.detach(),
        'group_2_is_metric_std': group_2_is_metric_std.detach(),
        'tinv_k1': K1,
        'tinv_k2': K2

    }
    return constraint_term, metrics

def welchs_ttest_ucb_constraint(is_reward, is_action, output_probs_a, sensitive_mask, log_lambda1, log_lambda2, constraint_kwargs, no_ucb=False, lagrange_exp=False):

    # Constraint Terms
    if(constraint_kwargs['disparity_type'] == 'reward'):
        group_1_is_metric = is_reward[sensitive_mask]
        group_2_is_metric = is_reward[~sensitive_mask]
    elif(constraint_kwargs['disparity_type'] == 'action'):
        group_1_is_metric = is_action[sensitive_mask]
        group_2_is_metric = is_action[~sensitive_mask]
    else:
        raise ValueError(f"Invalid disparity type provided: {constraint_kwargs['disparity_type']}")

    group_1_is_metric_mean, group_1_is_metric_std = group_1_is_metric.mean(), group_1_is_metric.std(correction=1)
    group_2_is_metric_mean, group_2_is_metric_std = group_2_is_metric.mean(), group_2_is_metric.std(correction=1)
    group_1_probs_a = output_probs_a[sensitive_mask]
    group_2_probs_a = output_probs_a[~sensitive_mask]

    # Expectation Term in constraint
    group_1_expectation_term = torch.mean(group_1_is_metric * torch.log(group_1_probs_a + 1e-8))
    group_2_expectation_term = torch.mean(group_2_is_metric * torch.log(group_2_probs_a + 1e-8))

    if no_ucb:
        fixed_lambda = constraint_kwargs['fixed_lambda']
        sign = torch.sign(group_1_is_metric_mean - group_2_is_metric_mean).detach()
        constraint_term = fixed_lambda * sign * (group_1_expectation_term - group_2_expectation_term)
        metrics = {
            'group_1_is_metric_mean': group_1_is_metric_mean.detach(),
            'group_1_is_metric_std': group_1_is_metric_std.detach(),
            'group_2_is_metric_mean': group_2_is_metric_mean.detach(),
            'group_2_is_metric_std': group_2_is_metric_std.detach(),
        }
        return constraint_term, metrics

    expectation_obs_term = (torch.exp(log_lambda1).detach().item()-torch.exp(log_lambda2).detach().item()) * (group_1_expectation_term - group_2_expectation_term)

    if lagrange_exp:
        constraint_term = expectation_obs_term
        with torch.no_grad():
            raw_g1_v_g2 = group_1_is_metric_mean - group_2_is_metric_mean
            raw_g2_v_g1 = group_2_is_metric_mean - group_1_is_metric_mean
        metrics = {
            'raw_g1_v_g2': raw_g1_v_g2.detach(),
            'raw_g2_v_g1': raw_g2_v_g1.detach(),
            'group_1_is_metric_mean': group_1_is_metric_mean.detach(),
            'group_1_is_metric_std': group_1_is_metric_std.detach(),
            'group_2_is_metric_mean': group_2_is_metric_mean.detach(),
            'group_2_is_metric_std': group_2_is_metric_std.detach(),
        }
        return constraint_term, metrics

    # Stdev Terms in constraint
    # With probability approximately 1-δ:
    # E[X] - E[Y] ≤ (x̄ - ȳ) + t_{1-δ, ν} · √(s_x²/n_x + s_y²/n_y)
    # where:
    # ν = (s_x²/n_x + s_y²/n_y)² / [(s_x²/n_x)²/(n_x-1) + (s_y²/n_y)²/(n_y-1)]
    satterthwaite_v_num = (group_1_is_metric_std**2 / constraint_kwargs['safety_data_grp_1_size']) + (group_2_is_metric_std**2 / constraint_kwargs['safety_data_grp_2_size'])
    satterthwaite_v_den = ((group_1_is_metric_std**2 / constraint_kwargs['safety_data_grp_1_size'])**2 / (constraint_kwargs['safety_data_grp_1_size'] - 1)) + ((group_2_is_metric_std**2 / constraint_kwargs['safety_data_grp_2_size'])**2 / (constraint_kwargs['safety_data_grp_2_size'] - 1))
    satterthwaite_v = (satterthwaite_v_num / (satterthwaite_v_den + 1e-8))
    K = stats.t.ppf(1 - (constraint_kwargs['fail_prob']/2.0), satterthwaite_v.item())
    K_scaled = K / torch.sqrt((group_1_is_metric_std**2 / constraint_kwargs['safety_data_grp_1_size']) + (group_2_is_metric_std**2 / constraint_kwargs['safety_data_grp_2_size']))
    group_1_norm_is_metric = (group_1_is_metric - group_1_is_metric_mean) / (group_1_is_metric_std + 1e-8)
    group_2_norm_is_metric = (group_2_is_metric - group_2_is_metric_mean) / (group_2_is_metric_std + 1e-8)
    group_1_stdev_term = torch.mean(group_1_norm_is_metric * group_1_is_metric * torch.log(group_1_probs_a + 1e-8))
    group_2_stdev_term = torch.mean(group_2_norm_is_metric * group_2_is_metric * torch.log(group_2_probs_a + 1e-8))
    stdev_obs_term = (torch.exp(log_lambda1).detach().item()+torch.exp(log_lambda2).detach().item()) * K_scaled * (((group_1_is_metric_std/constraint_kwargs['safety_data_grp_1_size'])*group_1_stdev_term) + ((group_2_is_metric_std/constraint_kwargs['safety_data_grp_2_size'])*group_2_stdev_term))

    constraint_term = expectation_obs_term + stdev_obs_term

    with torch.no_grad():
        ucb_g1_v_g2 = group_1_is_metric_mean - group_2_is_metric_mean + K*torch.sqrt((group_1_is_metric_std**2 / constraint_kwargs['safety_data_grp_1_size']) + (group_2_is_metric_std**2 / constraint_kwargs['safety_data_grp_2_size']))
        ucb_g2_v_g1 = group_2_is_metric_mean - group_1_is_metric_mean + K*torch.sqrt((group_1_is_metric_std**2 / constraint_kwargs['safety_data_grp_1_size']) + (group_2_is_metric_std**2 / constraint_kwargs['safety_data_grp_2_size']))

    metrics = {
        'ucb_g1_v_g2': ucb_g1_v_g2.detach(),
        'ucb_g2_v_g1': ucb_g2_v_g1.detach(),
        'group_1_is_metric_mean': group_1_is_metric_mean.detach(),
        'group_1_is_metric_std': group_1_is_metric_std.detach(),
        'group_2_is_metric_mean': group_2_is_metric_mean.detach(),
        'group_2_is_metric_std': group_2_is_metric_std.detach(),
        'tinv_satterthwaite': K.item(),
        'satterthwaite_v': satterthwaite_v.detach()

    }
    return constraint_term, metrics

def gather_is_rewards(policy, data, device, batch_size):
    """Collect IS-weighted returns, IS-weighted actions, and sensitive masks over a full dataset split."""
    all_is_rewards = []
    all_is_actions = []
    all_sensitive  = []

    policy.eval()
    with torch.no_grad():
        for b_sidx in range(0, len(data['context']), batch_size):
            batch_X = data['context'].iloc[b_sidx : b_sidx + batch_size]
            batch_a = data['actions'][['action', 'propensity']].iloc[b_sidx : b_sidx + batch_size]
            batch_r = data['reward'].iloc[b_sidx : b_sidx + batch_size]
            batch_s = data['sensitive'].iloc[b_sidx : b_sidx + batch_size]

            batch_X_tensor    = torch.tensor(batch_X.values, dtype=torch.float32).to(device)
            pi_b_actions      = torch.tensor(batch_a['action'].values, dtype=torch.long).unsqueeze(1).to(device)
            pi_b_action_probs = torch.tensor(batch_a['propensity'].values, dtype=torch.float32).unsqueeze(1).to(device)
            pi_b_rewards      = torch.tensor(batch_r.values, dtype=torch.float32).unsqueeze(1).to(device)

            output_probs_a = policy(batch_X_tensor).gather(1, pi_b_actions)
            is_reward = (output_probs_a / pi_b_action_probs) * pi_b_rewards  # (B, 1)
            is_action = (output_probs_a / pi_b_action_probs) * pi_b_actions.float()  # (B, 1)

            all_is_rewards.append(is_reward.squeeze(1))
            all_is_actions.append(is_action.squeeze(1))
            all_sensitive.append(torch.tensor(batch_s.values, dtype=torch.bool).squeeze(1).to(device))

    return torch.cat(all_is_rewards), torch.cat(all_is_actions), torch.cat(all_sensitive)  # all (N,)


def safety_test(policy, safety_data, device, constraint_kwargs, batch_size=128):

    all_is_rewards, all_is_actions, all_sensitive = gather_is_rewards(policy, safety_data, device, batch_size)

    if(constraint_kwargs['disparity_type'] == 'reward'):
        group_1_is_metric = all_is_rewards[all_sensitive]
        group_2_is_metric = all_is_rewards[~all_sensitive]
    elif(constraint_kwargs['disparity_type'] == 'action'):
        group_1_is_metric = all_is_actions[all_sensitive]
        group_2_is_metric = all_is_actions[~all_sensitive]
    else:
        raise ValueError(f"Invalid disparity type provided: {constraint_kwargs['disparity_type']}")

    group_1_is_metric_mean, group_1_is_metric_std = group_1_is_metric.mean(), group_1_is_metric.std(correction=1)
    group_2_is_metric_mean, group_2_is_metric_std = group_2_is_metric.mean(), group_2_is_metric.std(correction=1)
    
    if(constraint_kwargs['constraint_type'] == 'students-ttest'):
        K1 = stats.t.ppf(1 - (constraint_kwargs['fail_prob']/4.0), constraint_kwargs['safety_data_grp_1_size']-1)/math.sqrt(constraint_kwargs['safety_data_grp_1_size'])
        K2 = stats.t.ppf(1 - (constraint_kwargs['fail_prob']/4.0), constraint_kwargs['safety_data_grp_2_size']-1)/math.sqrt(constraint_kwargs['safety_data_grp_2_size'])

        ucb_g1_v_g2 = group_1_is_metric_mean - group_2_is_metric_mean + K1*group_1_is_metric_std + K2*group_2_is_metric_std
        ucb_g2_v_g1 = group_2_is_metric_mean - group_1_is_metric_mean + K1*group_1_is_metric_std + K2*group_2_is_metric_std

        return 'Safe' if (ucb_g1_v_g2.item() < constraint_kwargs['epsilon'] and ucb_g2_v_g1.item() < constraint_kwargs['epsilon']) else 'NSF'

    elif(constraint_kwargs['constraint_type'] == 'welchs-ttest'):
        satterthwaite_v_num = (group_1_is_metric_std**2 / constraint_kwargs['safety_data_grp_1_size']) + (group_2_is_metric_std**2 / constraint_kwargs['safety_data_grp_2_size'])
        satterthwaite_v_den = ((group_1_is_metric_std**2 / constraint_kwargs['safety_data_grp_1_size'])**2 / (constraint_kwargs['safety_data_grp_1_size'] - 1)) + ((group_2_is_metric_std**2 / constraint_kwargs['safety_data_grp_2_size'])**2 / (constraint_kwargs['safety_data_grp_2_size'] - 1))
        satterthwaite_v = (satterthwaite_v_num / (satterthwaite_v_den + 1e-8))
        K = stats.t.ppf(1 - (constraint_kwargs['fail_prob']/2.0), satterthwaite_v.item())

        ucb_g1_v_g2 = group_1_is_metric_mean - group_2_is_metric_mean + K*torch.sqrt((group_1_is_metric_std**2 / constraint_kwargs['safety_data_grp_1_size']) + (group_2_is_metric_std**2 / constraint_kwargs['safety_data_grp_2_size']))
        ucb_g2_v_g1 = group_2_is_metric_mean - group_1_is_metric_mean + K*torch.sqrt((group_1_is_metric_std**2 / constraint_kwargs['safety_data_grp_1_size']) + (group_2_is_metric_std**2 / constraint_kwargs['safety_data_grp_2_size']))

        return 'Safe' if (ucb_g1_v_g2.item() < constraint_kwargs['epsilon'] and ucb_g2_v_g1.item() < constraint_kwargs['epsilon']) else 'NSF'
    else:
        raise ValueError(f"Invalid constraint type provided: {constraint_kwargs['constraint_type']}")


def evaluate_on_test(policy, test_data, device, constraint_kwargs, batch_size=128):

    all_is_rewards, all_is_actions, all_sensitive = gather_is_rewards(policy, test_data, device, batch_size)

    if(constraint_kwargs['disparity_type'] == 'reward'):
        group_1_is_metric = all_is_rewards[all_sensitive]
        group_2_is_metric = all_is_rewards[~all_sensitive]
    elif(constraint_kwargs['disparity_type'] == 'action'):
        group_1_is_metric = all_is_actions[all_sensitive]
        group_2_is_metric = all_is_actions[~all_sensitive]
    else:
        raise ValueError(f"Invalid disparity type provided: {constraint_kwargs['disparity_type']}")

    constraint = torch.abs(group_1_is_metric.mean() - group_2_is_metric.mean())
    return 'Safe' if(constraint.item() < constraint_kwargs['epsilon']) else 'Unsafe'


def bound_propagation(is_reward, is_action, output_probs_a, sensitive_mask, constraint_kwargs):
    """
        This function returns the lower and upper bound of |E[.|A] - E[.|B]| 
    """
    # First compute LCB and UCB of E[.|A]
    # Constraint Terms
    if(constraint_kwargs['disparity_type'] == 'reward'):
        group_1_is_metric = is_reward[sensitive_mask]
        group_2_is_metric = is_reward[~sensitive_mask]
    elif(constraint_kwargs['disparity_type'] == 'action'):
        group_1_is_metric = is_action[sensitive_mask]
        group_2_is_metric = is_action[~sensitive_mask]
    else:
        raise ValueError(f"Invalid disparity type provided: {constraint_kwargs['disparity_type']}")

    group_1_is_metric_mean, group_1_is_metric_std = group_1_is_metric.mean(), group_1_is_metric.std(correction=1)
    group_2_is_metric_mean, group_2_is_metric_std = group_2_is_metric.mean(), group_2_is_metric.std(correction=1)
    group_1_probs_a = output_probs_a[sensitive_mask]
    group_2_probs_a = output_probs_a[~sensitive_mask]

    K1 = stats.t.ppf(1 - (constraint_kwargs['fail_prob']/4.0), constraint_kwargs['safety_data_grp_1_size']-1)/math.sqrt(constraint_kwargs['safety_data_grp_1_size'])
    K2 = stats.t.ppf(1 - (constraint_kwargs['fail_prob']/4.0), constraint_kwargs['safety_data_grp_2_size']-1)/math.sqrt(constraint_kwargs['safety_data_grp_2_size'])

    ucb_g1 = group_1_is_metric_mean + (K1 * group_1_is_metric_std)
    ucb_g2 = group_2_is_metric_mean + (K2 * group_2_is_metric_std)

    lcb_g1 = group_1_is_metric_mean - (K1 * group_1_is_metric_std)
    lcb_g2 = group_2_is_metric_mean - (K2 * group_2_is_metric_std)

    ucb_g1_diff_g2 = ucb_g1 - lcb_g2
    lcb_g1_diff_g2 = lcb_g1 - ucb_g2

    # Absolute value rule on d = E_g1 - E_g2, d ∈ [lcb_g1_diff_g2, ucb_g1_diff_g2]
    # If 0 ∈ [l, u]: |d| ∈ [0,          max(|l|, |u|)]
    # Otherwise:      |d| ∈ [min(|l|,|u|), max(|l|, |u|)]
    ucb_abs_diff = torch.max(torch.abs(lcb_g1_diff_g2), torch.abs(ucb_g1_diff_g2))
    lcb_abs_diff = torch.where(
        (lcb_g1_diff_g2 <= 0) & (ucb_g1_diff_g2 >= 0),
        torch.zeros(1, device=ucb_abs_diff.device),
        torch.min(torch.abs(lcb_g1_diff_g2), torch.abs(ucb_g1_diff_g2))
    )

    return lcb_abs_diff, ucb_abs_diff



