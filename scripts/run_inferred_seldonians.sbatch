#!/bin/bash
#SBATCH --job-name=inferred-seld
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --array=0-7
#SBATCH --output=slurm-inferred-seld-%x-%A_%a.out
#SBATCH --error=slurm-inferred-seld-%x-%A_%a.err

# Inferred setup: 2 Seldonians (usual method on inferred labels + inferred-attributes bound-prop).
# 52 seeds, same FRACS as before.
# Array 0-12 × 4 GPUs = 52 seeds.

module purge
module load anaconda3/2025.6
conda activate fair_bandits_v2

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export WANDB_MODE=offline

REPO="/scratch/gpfs/KOROLOVA/bt4811/rlc/Inferred-Bandits"
cd "$REPO"

FRACS=(0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 1.0)
ERR_RATES=(0.05 0.15)
INFERRED_PROP=0.5

for local_rank in 0 1 2 3; do
  SEED=$((SLURM_ARRAY_TASK_ID * 4 + local_rank))
  srun -N1 -n1 --exclusive --gres=gpu:1 bash -c "
    for ERR in ${ERR_RATES[*]}; do
      for FRAC in ${FRACS[*]}; do
        # 1) Usual Seldonian on inferred (noisy) sensitive labels
        python optimize.py \
          --config configs/default.yaml \
          --mode seldonian \
          --seed ${SEED} \
          --train_frac \${FRAC} \
          --inferred_proportion ${INFERRED_PROP} \
          --fpr \${ERR} \
          --fnr \${ERR} \
          --bandit_data_dir \"${REPO}/bandit_datasets\" \
          --output_dir \"${REPO}/outputs/inferred_exp/seldonian_inferred/err_\${ERR}/seed_${SEED}/frac_\${FRAC}\"

        # 2) Inferred-attributes Seldonian (bound propagation for noisy labels)
        python inferred-optimize-no-safe.py \
          --config configs/default_inferred_no_safe.yaml \
          --mode seldonian \
          --seed ${SEED} \
          --train_frac \${FRAC} \
          --inferred_proportion ${INFERRED_PROP} \
          --fpr \${ERR} \
          --fnr \${ERR} \
          --bandit_data_dir \"${REPO}/bandit_datasets\" \
          --output_dir \"${REPO}/outputs/inferred_exp/inferred_seldonian/err_\${ERR}/seed_${SEED}/frac_\${FRAC}\"
      done
    done
  " &
done

wait
