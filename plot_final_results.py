import argparse
import json
import math
import os

import numpy as np
import matplotlib.pyplot as plt


MODES = ("seldonian", "naive", "fixed_lambda", "lagrange_exp")
COLORS = {
    "seldonian": "#2ca02c",
    "naive": "#1f77b4",
    "fixed_lambda": "#ff7f0e",
    "lagrange_exp": "#9467bd",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, default="./outputs/final_results.json")
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/plots")
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--x_axis", type=str, default="fraction", choices=["fraction", "samples"])
    parser.add_argument("--train_size_total", type=int, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--sensitive_attribute", type=str, default=None)
    parser.add_argument("--behavior_policy_type", type=str, default=None)
    parser.add_argument("--constraint_type", type=str, default=None)
    parser.add_argument("--disparity_type", type=str, default=None)
    return parser.parse_args()


def deep_merge(a, b):
    for key, val in b.items():
        if key in a and isinstance(a[key], dict) and isinstance(val, dict):
            deep_merge(a[key], val)
        else:
            a[key] = val
    return a


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Handle concatenated JSON objects or JSONL
        decoder = json.JSONDecoder()
        idx = 0
        merged = {}
        while idx < len(content):
            while idx < len(content) and content[idx].isspace():
                idx += 1
            if idx >= len(content):
                break
            obj, next_idx = decoder.raw_decode(content, idx)
            if isinstance(obj, dict):
                deep_merge(merged, obj)
            idx = next_idx
        return merged


def matches_filters(record, args):
    if args.dataset_name and record.get("dataset_name") != args.dataset_name:
        return False
    if args.sensitive_attribute and record.get("sensitive_attribute") != args.sensitive_attribute:
        return False
    if args.behavior_policy_type and record.get("behavior_policy_type") != args.behavior_policy_type:
        return False
    if args.constraint_type and record.get("constraint_type") != args.constraint_type:
        return False
    if args.disparity_type and record.get("disparity_type") != args.disparity_type:
        return False
    return True


def recompute_flags(record, epsilon):
    safety = record.get("safety", {})
    test = record.get("test", {})
    ucb_g1_v_g2 = safety.get("ucb_g1_v_g2")
    ucb_g2_v_g1 = safety.get("ucb_g2_v_g1")
    constraint = test.get("constraint")

    safety_safe = (ucb_g1_v_g2 is not None and ucb_g2_v_g1 is not None and
                   ucb_g1_v_g2 < epsilon and ucb_g2_v_g1 < epsilon)
    test_safe = (constraint is not None and constraint < epsilon)
    return safety_safe, test_safe


def get_mean_reward(record):
    test = record.get("test", {})
    if "mean_is_reward" in test:
        return test["mean_is_reward"]
    return record.get("test_mean_is_reward")


def x_value(frac, records, args):
    if args.x_axis == "fraction":
        return frac
    base = None
    # Prefer per-record cap if present
    if records:
        max_ex = records[0].get("max_ex_train")
        if max_ex is not None:
            base = max_ex
    if base is None:
        base = args.train_size_total
    if base is None:
        raise ValueError("--train_size_total is required when --x_axis samples and max_ex_train is not set")
    return frac * base


def plot_with_band(x_vals, means, stds, label, color):
    means = np.array(means, dtype=float)
    stds = np.array(stds, dtype=float)
    x_vals = np.array(x_vals, dtype=float)
    plt.plot(x_vals, means, marker="o", linewidth=2, markersize=4, color=color, label=label)
    plt.fill_between(x_vals, means - stds, means + stds, color=color, alpha=0.2, linewidth=0)


def collect_by_mode_and_frac(data, args):
    grouped = {}
    for mode, frac_map in data.items():
        if mode not in MODES:
            continue
        for frac_str, seed_map in frac_map.items():
            try:
                frac = float(frac_str)
            except ValueError:
                continue
            for seed_str, record in seed_map.items():
                if not matches_filters(record, args):
                    continue
                grouped.setdefault(mode, {}).setdefault(frac, []).append(record)
    return grouped


def plot_reward(grouped, epsilon, output_path, args):
    plt.figure(figsize=(8, 5))
    for mode in MODES:
        if mode not in grouped:
            continue
        fracs = sorted(grouped[mode].keys())
        means = []
        stds = []
        x_vals = []
        for frac in fracs:
            records = grouped[mode][frac]
            rewards = []
            for record in records:
                safety_safe, _ = recompute_flags(record, epsilon)
                if (mode == "seldonian" or mode == "cma") and not safety_safe:
                    continue
                reward = get_mean_reward(record)
                if reward is not None:
                    rewards.append(reward)
            if rewards:
                means.append(float(np.mean(rewards)))
                stds.append(float(np.std(rewards, ddof=0)))
            else:
                means.append(math.nan)
                stds.append(math.nan)
            x_vals.append(x_value(frac, records, args))
        plot_with_band(x_vals, means, stds, mode, COLORS.get(mode))

    plt.xlabel("Training samples" if args.x_axis == "samples" else "Training fraction")
    plt.ylabel("Mean IS reward")
    plt.title(f"Mean reward vs data fraction (epsilon={epsilon})")
    plt.xscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_retention(grouped, epsilon, output_path, args):
    plt.figure(figsize=(8, 5))
    for mode in MODES:
        if mode not in grouped:
            continue
        fracs = sorted(grouped[mode].keys())
        rates = []
        stds = []
        x_vals = []
        for frac in fracs:
            records = grouped[mode][frac]
            if not records:
                rates.append(math.nan)
                stds.append(math.nan)
                continue
            vals = []
            if mode == "seldonian" or mode == "cma":
                for record in records:
                    safety_safe, _ = recompute_flags(record, epsilon)
                    vals.append(1.0 if safety_safe else 0.0)
            else:
                vals = [1.0] * len(records)
            rates.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=0)))
            x_vals.append(x_value(frac, records, args))
        plot_with_band(x_vals, rates, stds, mode, COLORS.get(mode))

    plt.xlabel("Training samples" if args.x_axis == "samples" else "Training fraction")
    plt.ylabel("Retention rate")
    plt.title(f"Retention rate vs data fraction (epsilon={epsilon})")
    plt.ylim(-0.05, 1.05)
    plt.xscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_error_rate(grouped, epsilon, output_path, args):
    plt.figure(figsize=(8, 5))
    for mode in MODES:
        if mode not in grouped:
            continue
        fracs = sorted(grouped[mode].keys())
        rates = []
        stds = []
        x_vals = []
        for frac in fracs:
            records = grouped[mode][frac]
            if not records:
                rates.append(math.nan)
                stds.append(math.nan)
                continue
            vals = []
            if mode == "seldonian" or mode == "cma":
                # Unsafe on test but passed safety (false-safe)
                for record in records:
                    safety_safe, test_safe = recompute_flags(record, epsilon)
                    vals.append(1.0 if (safety_safe and not test_safe) else 0.0)
            else:
                for record in records:
                    _, test_safe = recompute_flags(record, epsilon)
                    vals.append(1.0 if not test_safe else 0.0)
            rates.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=0)))
            x_vals.append(x_value(frac, records, args))
        plot_with_band(x_vals, rates, stds, mode, COLORS.get(mode))

    plt.xlabel("Training samples" if args.x_axis == "samples" else "Training fraction")
    plt.ylabel("Error rate")
    plt.title(f"Error rate vs data fraction (epsilon={epsilon})")
    plt.ylim(-0.05, 1.05)
    plt.xscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 120,
    })
    data = load_results(args.results_path)

    grouped = collect_by_mode_and_frac(data, args)
    os.makedirs(args.output_dir, exist_ok=True)

    eps_tag = str(args.epsilon).replace(".", "p")
    reward_path = os.path.join(args.output_dir, f"reward_mean_eps{eps_tag}_{args.version}.png")
    retention_path = os.path.join(args.output_dir, f"retention_eps{eps_tag}_{args.version}.png")
    error_path = os.path.join(args.output_dir, f"error_rate_eps{eps_tag}_{args.version}.png")

    plot_reward(grouped, args.epsilon, reward_path, args)
    plot_retention(grouped, args.epsilon, retention_path, args)
    plot_error_rate(grouped, args.epsilon, error_path, args)

    print("Saved:")
    print(f"  {reward_path}")
    print(f"  {retention_path}")
    print(f"  {error_path}")


if __name__ == "__main__":
    main()
