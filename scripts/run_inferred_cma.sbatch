#!/bin/bash
#SBATCH --job-name=inferred-cma
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --array=0-7
#SBATCH --output=slurm-inferred-cma-%x-%A_%a.out
#SBATCH --error=slurm-inferred-cma-%x-%A_%a.err

# Same experimental batch as inferred Seldonians: CMA-ES on inferred (noisy) sensitive labels
# using the usual method. 52 seeds, same FRACS. Output under inferred_exp/ for comparison.
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
        python optimize_cma.py \
          --config configs/default_cma_dis_act.yaml \
          --seed ${SEED} \
          --train_frac \${FRAC} \
          --inferred_proportion ${INFERRED_PROP} \
          --fpr \${ERR} \
          --fnr \${ERR} \
          --n_jobs ${SLURM_CPUS_PER_TASK} \
          --bandit_data_dir \"${REPO}/bandit_datasets\" \
          --output_dir \"${REPO}/outputs/inferred_exp/cma_inferred/err_\${ERR}/seed_${SEED}/frac_\${FRAC}\"
      done
    done
  " &
done

wait
