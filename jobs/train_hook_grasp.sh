#!/bin/bash -l
#SBATCH --job-name=hook_grasp_train
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=/home/dezen/pds-lis/slurm_logs/%x_%j.out
#SBATCH --error=/home/dezen/pds-lis/slurm_logs/%x_%j.err

set -euo pipefail

# ---- Parameters (script arguments; defaults = full recipe from the README)
NUM_ENVS=${1:-512}
MAX_ITERS=${2:-20000}
RUN_NAME=${3:-hook_grasp}
SEED=${4:-42}
MINI_BATCHES=${5:-}   # empty = value from agents/rsl_rl_ppo_cfg.py (4)

# ---- Paths
PDS=$HOME/pds-lis
SIF=${SIF:-$PDS/containers/isaaclab_3.0.0-beta2.sif}   # override: SIF=... sbatch jobs/train_hook_grasp.sh
REPO=$PDS/repos/MARL_cooperative_aerial_manipulation_ext
SLURM_OUT=$PDS/slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out

echo "Start job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "num_envs=$NUM_ENVS  max_iterations=$MAX_ITERS  run_name=$RUN_NAME  seed=$SEED  num_mini_batches=${MINI_BATCHES:-cfg}"
echo "git commit: $(git -C "$REPO" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# GPU utilisation + memory every 5 min, to check whether the GPU is saturated (num_envs choice)
GPU_LOG=$PDS/slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu.csv
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 300 > "$GPU_LOG" &
GPU_MON_PID=$!

# train.py writes logs/ relative to the current directory
cd "$REPO"

apptainer exec --nv \
    -B "$PDS/isaac_sim_cache/data:/isaac-sim/kit/data" \
    -B "$PDS/isaac_sim_cache/cache:/isaac-sim/kit/cache" \
    -B "$PDS/isaac_sim_cache/logs:/isaac-sim/kit/logs" \
    -B "$REPO:/home/fuda/PycharmProjects/MARL_cooperative_aerial_manipulation_ext" \
    --env ACCEPT_EULA=Y \
    --env OMNI_KIT_ACCEPT_EULA=YES \
    --env PYTHONUNBUFFERED=1 \
    --env PYTHONPATH="$PDS/python_packages:$REPO/exts/MARL_mav_carry_ext" \
    "$SIF" \
    /workspace/isaaclab/_isaac_sim/python.sh scripts/hook_grasp/train.py \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$MAX_ITERS" \
        --run_name "$RUN_NAME" \
        --seed "$SEED" \
        ${MINI_BATCHES:+--num_mini_batches "$MINI_BATCHES"} \
    2>&1

kill "$GPU_MON_PID" 2>/dev/null || true

# train.py always exits with code 0 (os._exit): detect errors from the log
if grep -q "Traceback" "$SLURM_OUT"; then
    echo "!!! Python error detected in $SLURM_OUT"
    exit 1
fi
echo "End at $(date)"
