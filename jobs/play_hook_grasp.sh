#!/bin/bash -l
#SBATCH --job-name=hook_grasp_play
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/home/dezen/pds-lis/slurm_logs/%x_%j.out
#SBATCH --error=/home/dezen/pds-lis/slurm_logs/%x_%j.err
set -euo pipefail
CKPT=${1:?usage: sbatch jobs/play_hook_grasp.sh <checkpoint relative to the repo> [tag] [video: traj|isaac|both]}
TAG=${2:-hook_grasp}
# traj = matplotlib animation (works on V100s); isaac = RTX camera video (black frames on V100s); both
VIZ=${3:-traj}
PDS=$HOME/pds-lis
SIF=${SIF:-$PDS/containers/isaaclab_3.0.0-beta2.sif}
REPO=$PDS/repos/MARL_cooperative_aerial_manipulation_ext
cd "$REPO"
if [ ! -f "$CKPT" ]; then
    echo "!!! checkpoint not found: $REPO/$CKPT"
    exit 1
fi
apptainer exec --nv \
    -B "$PDS/isaac_sim_cache/data:/isaac-sim/kit/data" \
    -B "$PDS/isaac_sim_cache/cache:/isaac-sim/kit/cache" \
    -B "$PDS/isaac_sim_cache/logs:/isaac-sim/kit/logs" \
    -B "$REPO:/home/fuda/PycharmProjects/MARL_cooperative_aerial_manipulation_ext" \
    --env ACCEPT_EULA=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONUNBUFFERED=1 \
    --env PYTHONPATH="$PDS/python_packages:$REPO/exts/MARL_mav_carry_ext" \
    "$SIF" \
    /workspace/isaaclab/_isaac_sim/python.sh scripts/hook_grasp/play.py \
        --headless --checkpoint "$CKPT" --tag "$TAG" --video_type "$VIZ" --steps 900 2>&1
# play.py always exits with code 0 (os._exit): detect errors from the log
SLURM_OUT=$PDS/slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out
if grep -q "Traceback" "$SLURM_OUT"; then
    echo "!!! Python error detected in $SLURM_OUT (no output generated)"
    exit 1
fi
echo "Outputs: $REPO/videos/${TAG}_rollout.npz, ${TAG}_summary.txt, ${TAG}_viewer.html (interactive 3D viewer, open in a browser) and the video(s) ${TAG}_traj.mp4 / ${TAG}_isaac.mp4"
