#!/bin/bash -l
#SBATCH --job-name=hook_grasp_eval_run
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/home/dezen/pds-lis/slurm_logs/%x_%j.out
#SBATCH --error=/home/dezen/pds-lis/slurm_logs/%x_%j.err
# Evaluates several checkpoints of a run, automatically picking its latest timestamped folder.
#   sbatch jobs/eval_run_hook_grasp.sh <run_name> [iter ...]      (iter = number or "last"; default: last)
# Chained after a training job (afterany: also evaluates what exists if training hit the time limit):
#   sbatch --dependency=afterany:<jobid_train> jobs/eval_run_hook_grasp.sh accel_4096_5k 3000 4000 last
# About 4-5 min per checkpoint (including Isaac startup): 3 checkpoints fit in the 1 h debug QOS.
set -uo pipefail   # no -e: one failing checkpoint must not prevent evaluating the next ones
RUN=${1:?usage: sbatch jobs/eval_run_hook_grasp.sh <run_name> [iter ...]}
shift
ITERS=("${@:-last}")
STEPS=${STEPS:-4000}        # same values as the baseline evals (eval_hook_grasp.sh)
NUM_ENVS=${NUM_ENVS:-256}
PDS=$HOME/pds-lis
SIF=${SIF:-$PDS/containers/isaaclab_3.0.0-beta2.sif}
REPO=$PDS/repos/MARL_cooperative_aerial_manipulation_ext
SLURM_OUT=$PDS/slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out
echo "Start job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "git commit: $(git -C "$REPO" rev-parse --short HEAD)"
cd "$REPO"

# latest timestamped folder of the run (YYYY-MM-DD_HH-MM-SS names sort chronologically)
RUN_DIR=$(ls -d "logs/rsl_rl/$RUN"/*/ 2>/dev/null | sort | tail -1)
RUN_DIR=${RUN_DIR%/}
if [ -z "$RUN_DIR" ]; then
    echo "!!! no folder logs/rsl_rl/$RUN/<timestamp>/"
    exit 1
fi
echo "run_dir=$RUN_DIR  iters=${ITERS[*]}  steps=$STEPS  num_envs=$NUM_ENVS"

RESULTS=()
for IT in "${ITERS[@]}"; do
    if [ "$IT" = "last" ]; then
        CKPT=$(ls "$RUN_DIR"/model_*.pt 2>/dev/null | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
    else
        CKPT="$RUN_DIR/model_${IT}.pt"
    fi
    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo "!!! checkpoint not found for iter=$IT, skipped"
        continue
    fi
    CK_IT=$(basename "$CKPT" .pt | sed 's/model_//')
    OUT="videos/${RUN}_${CK_IT}_eval.json"
    echo "===== eval $CKPT -> $OUT ($(date +%H:%M:%S))"
    # only one Kit app per process: one python call per checkpoint
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
            --out "$OUT" 2>&1
    RESULTS+=("$OUT")
done

echo
echo "==================== SUMMARY $RUN ===================="
for OUT in "${RESULTS[@]}"; do
    if [ -f "$OUT" ]; then
        echo "$OUT: $(grep -E '"(episodes|catch_rate|hold_success_given_caught)"' "$OUT" | tr -d ' \n')"
    else
        echo "$OUT: no result (see the log above)"
    fi
done
# eval.py always exits with code 0 (os._exit): detect errors from the log
if grep -q "Traceback" "$SLURM_OUT"; then
    echo "!!! Python error detected in $SLURM_OUT"
    exit 1
fi
echo "End at $(date)"
