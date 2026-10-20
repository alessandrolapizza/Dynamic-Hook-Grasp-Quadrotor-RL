#!/bin/bash -l
#SBATCH --job-name=hook_grasp_eval
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/dezen/pds-lis/slurm_logs/%x_%j.out
#SBATCH --error=/home/dezen/pds-lis/slurm_logs/%x_%j.err
set -euo pipefail
CKPT=${1:?usage: sbatch jobs/eval_hook_grasp.sh <checkpoint relative to the repo> [tag] [steps] [num_envs]}
TAG=${2:-hook_grasp}
STEPS=${3:-4000}
NUM_ENVS=${4:-256}
PDS=$HOME/pds-lis
SIF=${SIF:-$PDS/containers/isaaclab_3.0.0-beta2.sif}
REPO=$PDS/repos/MARL_cooperative_aerial_manipulation_ext
SLURM_OUT=$PDS/slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out
echo "Start job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "checkpoint=$CKPT  tag=$TAG  steps=$STEPS  num_envs=$NUM_ENVS"
cd "$REPO"
apptainer exec --nv \
    -B "$PDS/isaac_sim_cache/data:/isaac-sim/kit/data" \
    -B "$PDS/isaac_sim_cache/cache:/isaac-sim/kit/cache" \
    -B "$PDS/isaac_sim_cache/logs:/isaac-sim/kit/logs" \
    -B "$REPO:/home/fuda/PycharmProjects/MARL_cooperative_aerial_manipulation_ext" \
    --env ACCEPT_EULA=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONUNBUFFERED=1 \
    --env PYTHONPATH="$PDS/python_packages:$REPO/exts/MARL_mav_carry_ext" \
    "$SIF" \
    /workspace/isaaclab/_isaac_sim/python.sh scripts/hook_grasp/eval.py \
        --headless --checkpoint "$CKPT" --steps "$STEPS" --num_envs "$NUM_ENVS" \
        --out "videos/${TAG}_eval.json" 2>&1
# eval.py always exits with code 0 (os._exit): detect errors from the log
if grep -q "Traceback" "$SLURM_OUT"; then
    echo "!!! Python error detected in $SLURM_OUT"
    exit 1
fi
echo "Result: $REPO/videos/${TAG}_eval.json"
