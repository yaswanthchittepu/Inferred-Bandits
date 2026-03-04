import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import math
import os, sys, ipdb
import wandb
from tqdm import tqdm

class PolicyNet(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Return probabilities (Policy pi_theta)
        return F.softmax(self.output(x), dim=-1)

def load_data(path):
    columns = [
        'id', 'age', 'gender', 'education', 'country', 'ethnicity', 'nscore', 
        'escore', 'oscore', 'ascore', 'cscore', 'impulsive', 'ss', 'alcohol', 
        'amphet', 'amyl', 'benzos', 'caff', 'cannabis', 'choc', 'coke', 'crack', 
        'ecstasy', 'heroin', 'ketamine', 'legalh', 'lsd', 'meth', 'mushrooms',
        'nicotine', 'semer', 'vsa'
    ]
    drug_col_names = columns[13:]
    class_meanings = {
        'CL0': 'Never Used', 'CL1': 'Used over a Decade Ago', 
        'CL2': 'Used in Last Decade', 
        'CL3': 'Used in Last Year', 
        'CL4': 'Used in Last Month', 'CL5': 'Used in Last Week', 
        'CL6': 'Used in Last Day'
    }

    # Define numerical ranks for comparison (CL6 is the 'maximum' recency)
    class_rank = {
        'CL0': 0, 'CL1': 1, 'CL2': 2, 'CL3': 3, 'CL4': 4, 'CL5': 5, 'CL6': 6
    }
    class_mappings = {
        'CL0': '10 years', 'CL1': '10 years', 
        'CL2': '5 years', 
        'CL3': 'past year', 
        'CL4': 'past month', 'CL5': 'past month', 'CL6': 'past month'
    }
    
    data = pd.read_csv(path, names=columns, index_col='id')
    # Gender: Original -0.48246 is Male, 0.48246 is Female
    data['gender_label'] = np.where(data['gender'] < 0, 'M', 'F')

    # Education: Values < 0.45468 are below University degree
    data['education_label'] = np.where(data['education'] >= 0.45468, 'above college', 'below college')

    # Get overall drug use
    # Find the maximum usage across ALL drug columns for each row
    data['overall_drug_use'] = data[drug_col_names].apply(lambda row: max(row, key=lambda x: class_rank[x]), axis=1)
    
    is_verified = data.apply(lambda row: row['overall_drug_use'] in row[drug_col_names].values, axis=1)

    if not is_verified.all():
        print(f"Warning: {(~is_verified).sum()} rows failed verification.")
    else:
        print("Verification passed: All overall_drug_use labels exist in source drug columns.")

    # This is for consistency with Group Fairness paper. 
    # But can be skipped as their choice is arbitrary
    # Relabel Classes for all drugs
    for drug in drug_col_names+['overall_drug_use']:
        data[drug] = data[drug].map(class_mappings)

    drug_value_counts_result = data[drug_col_names].apply(pd.Series.value_counts)

    data.drop(columns=drug_col_names, inplace=True)  # Drop original drug columns, keep only overall_drug_use

    return data

def create_bandit_data(df, target_col, K=4):
    """
    df: cleaned dataframe
    target_col: the column containing ground truth bins
    K: number of times to use each row as a context
    """
    classes = ['10 years', '5 years', 'past year', 'past month']
    mapping = {label: i for i, label in enumerate(classes)}
    df[f'{target_col}_label'] = df[target_col].map(mapping)

    n_actions = len(classes)
    
    # 1. Repeat each row K times to expand the dataset
    # This creates a larger context pool for the policy to act upon
    df_expanded = df.loc[df.index.repeat(K)].reset_index(drop=True)
    n_samples = len(df_expanded)
    
    # 2. Generate random actions for the expanded pool
    # pi_b(a|x) = 1/K_actions
    action_indices = np.random.choice(range(n_actions), size=n_samples)
    
    # 3. Map indices to labels for reward check
    chosen_labels = np.array([classes[i] for i in action_indices])
    
    # 4. Calculate Reward (+1 if correct, -1 otherwise)
    # Each trial for the same context might get a different action/reward
    rewards = np.where(chosen_labels == df_expanded[target_col], 1, -1)
    
    # 5. Final Bandit Dataset
    bandit_df = df_expanded.copy()
    bandit_df['action'] = action_indices
    bandit_df['reward'] = rewards
    bandit_df['propensity'] = 1.0 / n_actions

    bandit_df_shuffled = bandit_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return bandit_df_shuffled


if __name__ == "__main__":

    num_epochs = 100
    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epsilon = 0.2
    fail_prob = 0.1
    log_lambda_max = 2
    eval_freq = 25

    wandb.init(
        project="group-fair-bandits",
        name="seldonian-lagrangian",
        config={
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "epsilon": epsilon,
            "fail_prob": fail_prob,
            "log_lambda_max": log_lambda_max,
            "eval_freq": eval_freq,
        },
    )


    data = load_data('./data-uci/drug+consumption+quantified.zip')
    bandit_data = create_bandit_data(data, 'overall_drug_use', K=4)

    true_labels = bandit_data[['overall_drug_use', 'overall_drug_use_label']]
    bandit_data.drop(columns=['overall_drug_use', 'overall_drug_use_label'], inplace=True)  # Drop the target column, keep only context and bandit info

    # 1. Define the full feature set (Context) and the Bandit data (Action, Reward, Propensity)
    # Ensure sensitive attributes are excluded from 'X' if you want a 'blind' policy
    X_bandit = bandit_data.drop(columns=['action', 'reward', 'propensity'], errors='ignore')
    y_bandit = bandit_data[['action', 'reward', 'propensity']]

    # 2. First Split: Train vs. (Safety + Test)
    # 70% Train, 30% for the rest
    X_train, X_rem, y_train, y_rem, labels_train, labels_rem = train_test_split(
        X_bandit, y_bandit, true_labels, test_size=0.3, random_state=42, stratify=true_labels
    )

    # 3. Second Split: Safety vs. Test
    # Split the remaining 30% in half to get 15% Safety and 15% Test
    X_safety, X_test, y_safety, y_test, labels_safety, labels_test = train_test_split(
        X_rem, y_rem, labels_rem, test_size=0.5, random_state=42, stratify=labels_rem
    )

    print(f"Train size:  {len(X_train)}")
    print(f"Safety size: {len(X_safety)}")
    print(f"Test size:   {len(X_test)}")

    # REMEBER TO DROP gender_label AND education_label before training.
    drop_columns_at_train = ['gender_label', 'education_label']

    # # 2. Setup Hyperparameters
    input_dim = X_train.shape[1] - len(drop_columns_at_train)  # Number of context features
    n_actions = 4
    policy = PolicyNet(input_dim, n_actions).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    # Lagrangian multipliers in log space (exp ensures positivity)
    log_lambda1 = torch.zeros(1, requires_grad=True, device=device)
    log_lambda2 = torch.zeros(1, requires_grad=True, device=device)
    lambda_optimizer = torch.optim.Adam([log_lambda1, log_lambda2], lr=5e-2)

    safety_data_size = len(X_safety)
    safety_data_grp_1_size = X_safety[X_safety['gender_label'] == 'M'].shape[0]
    safety_data_grp_2_size = safety_data_size - safety_data_grp_1_size

    epoch_iter = tqdm(range(num_epochs), desc="Training")
    for epoch in epoch_iter:
        # 1. Generate shuffled indices for this epoch
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)

        # 2. Reorder data using the shuffled indices
        # (Assuming these are pandas DataFrames/Series or NumPy arrays)
        X_shuffled = X_train[indices] if isinstance(X_train, np.ndarray) else X_train.iloc[indices]
        y_shuffled = y_train[indices] if isinstance(y_train, np.ndarray) else y_train.iloc[indices]
        labels_shuffled = labels_train[indices] if isinstance(labels_train, np.ndarray) else labels_train.iloc[indices]

        # 3. Iterate over the data in batches
        batch_iter = tqdm(enumerate(range(0, len(X_shuffled), batch_size)),
                          total=len(X_shuffled) // batch_size,
                          desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for i, b_sidx in batch_iter:
            
            step_idx = (len(X_shuffled)//batch_size)*epoch + i
            # Slice the batch
            batch_X = X_shuffled.iloc[b_sidx : b_sidx + batch_size]
            batch_y = y_shuffled.iloc[b_sidx : b_sidx + batch_size]
            batch_labels = labels_shuffled.iloc[b_sidx : b_sidx + batch_size]
            
            # --- Your Training Logic Here ---
            # Mask, dim => (B,)
            gender_mask = torch.tensor((batch_X['gender_label'] == 'M').values)
            education_mask = torch.tensor((batch_X['education_label'] == 'above college').values)

            batch_X = batch_X.drop(columns=drop_columns_at_train, errors='ignore')  # Drop sensitive attributes for training
            
            # X => (B, D), action => (B,1), reward => (B,1), propensity => (B,1)
            batch_X_tensor = torch.tensor(batch_X.values, dtype=torch.float32).to(device)  # Add a dimension for batch processing
            pi_b_actions = torch.tensor(batch_y['action'].values, dtype=torch.long).unsqueeze(1).to(device)
            pi_b_rewards = torch.tensor(batch_y['reward'].values, dtype=torch.float32).unsqueeze(1).to(device)
            pi_b_propensity = torch.tensor(batch_y['propensity'].values, dtype=torch.float32).unsqueeze(1).to(device)
            true_batch_labels = torch.tensor(batch_labels['overall_drug_use_label'].values, dtype=torch.long).to(device)
            
            output_probs = policy(batch_X_tensor)
            # output_probs_a => (B,1) - the probability of the action that was actually taken by our policy
            output_probs_a = output_probs.gather(1, pi_b_actions)

            # Objective Term
            is_reward = (output_probs_a.detach() / pi_b_propensity) * pi_b_rewards
            reward_obj_term = torch.mean(is_reward * torch.log(output_probs_a + 1e-8))  # Add small epsilon for stability

            # Constraint Terms
            group_1_is_reward = is_reward[gender_mask]
            group_2_is_reward = is_reward[~gender_mask]

            if(len(group_1_is_reward) == 0 or len(group_2_is_reward) == 0):
                print(f"Warning: One of the groups has no samples in this batch. Skipping fairness constraint for this batch.")
                continue

            group_1_is_reward_mean, group_1_is_reward_std = group_1_is_reward.mean(), group_1_is_reward.std(correction=1)
            group_2_is_reward_mean, group_2_is_reward_std = group_2_is_reward.mean(), group_2_is_reward.std(correction=1)
            group_1_probs_a = output_probs_a[gender_mask]
            group_2_probs_a = output_probs_a[~gender_mask]

            # Expectation Term in constraint
            group_1_expectation_term = torch.mean(group_1_is_reward * torch.log(group_1_probs_a + 1e-8))
            group_2_expectation_term = torch.mean(group_2_is_reward * torch.log(group_2_probs_a + 1e-8))
            expectation_obs_term = (torch.exp(log_lambda1).detach().item()-torch.exp(log_lambda2).detach().item()) * (group_1_expectation_term - group_2_expectation_term)

            # Stdev Terms in constraint
            group_1_norm_is_rewards = (group_1_is_reward - group_1_is_reward_mean) / (group_1_is_reward_std + 1e-8)
            group_2_norm_is_rewards = (group_2_is_reward - group_2_is_reward_mean) / (group_2_is_reward_std + 1e-8)
            group_1_stdev_term = torch.mean(group_1_norm_is_rewards * group_1_is_reward * torch.log(group_1_probs_a + 1e-8))
            group_2_stdev_term = torch.mean(group_2_norm_is_rewards * group_2_is_reward * torch.log(group_2_probs_a + 1e-8))
            K1 = stats.t.ppf(1 - (fail_prob/4.0), safety_data_grp_1_size-1)/math.sqrt(safety_data_grp_1_size)
            K2 = stats.t.ppf(1 - (fail_prob/4.0), safety_data_grp_2_size-1)/math.sqrt(safety_data_grp_2_size)
            stdev_obs_term = (torch.exp(log_lambda1).detach().item()+torch.exp(log_lambda2).detach().item()) * ((K1*group_1_stdev_term) + (K2*group_2_stdev_term))    
            
            # Lambda update step
            with torch.no_grad():
                ucb_g1_v_g2 = group_1_is_reward_mean - group_2_is_reward_mean + K1*group_1_is_reward_std + K2*group_2_is_reward_std
                ucb_g2_v_g1 = group_2_is_reward_mean - group_1_is_reward_mean + K1*group_1_is_reward_std + K2*group_2_is_reward_std
            lambda_optimizer.zero_grad()
            lambda_loss = -((ucb_g1_v_g2.detach() - epsilon) * log_lambda1.exp()) - ((ucb_g2_v_g1.detach() - epsilon) * log_lambda2.exp())
            lambda_loss.backward()
            lambda_optimizer.step()
            with torch.no_grad():
                log_lambda1.clamp_(max=log_lambda_max)
                log_lambda2.clamp_(max=log_lambda_max)

            optimizer.zero_grad()
            overall_obj = reward_obj_term - expectation_obs_term - stdev_obs_term
            # overall_obj = reward_obj_term
            loss = -overall_obj

            loss.backward()
            optimizer.step()

            log_dict = {
                "loss": loss.detach().item(),
                "lambda1": log_lambda1.detach().exp().item(),
                "lambda2": log_lambda2.detach().exp().item(),
                "is_reward_mean": is_reward.mean().item(),
                "ucb_g1_v_g2": ucb_g1_v_g2.item(),
                "ucb_g2_v_g1": ucb_g2_v_g1.item(),
                "mean_g1": group_1_is_reward_mean.item(),
                "mean_g2": group_2_is_reward_mean.item(),
                "stdev_g1": group_1_is_reward_std.item(),
                "stdev_g2": group_2_is_reward_std.item(),
            }

            if(step_idx % eval_freq == 0):
                with torch.no_grad():
                    output_probs_eval = policy(batch_X_tensor)
                    predicted_actions_eval = torch.argmax(output_probs_eval, dim=1)  # Get the action with highest probability for each context
                    match_eval = predicted_actions_eval.squeeze() == true_batch_labels.squeeze()
                    rewards_eval = torch.where(match_eval, torch.tensor(1.0), torch.tensor(-1.0))
                    mean_reward_eval = rewards_eval.mean()

                    grp_1_rewards_eval = rewards_eval[gender_mask]
                    grp_2_rewards_eval = rewards_eval[~gender_mask]
                    constraint_eval = torch.abs(grp_1_rewards_eval.mean() - grp_2_rewards_eval.mean()) - epsilon
                log_dict["mean_reward_eval_on_policy"] = mean_reward_eval.item()
                log_dict["constraint_eval_on_policy"] = constraint_eval.item()

            wandb.log(log_dict, step=step_idx)
            
        print(f"Epoch {epoch+1}/{num_epochs} complete.")