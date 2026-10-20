#!/bin/bash -l
#SBATCH --job-name=build_isaaclab_sif
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/dezen/pds-lis/slurm_logs/%x_%j.out
#SBATCH --error=/home/dezen/pds-lis/slurm_logs/%x_%j.err

set -euo pipefail

echo "Start job ${SLURM_JOB_ID} on $(hostname) at $(date)"

OUT_DIR=$HOME/pds-lis/containers

# Layer cache on scratch, build directory on the node's fast local disk
export APPTAINER_CACHEDIR=/scratch/izar/$USER/apptainer/cache
export APPTAINER_TMPDIR=/tmp/$USER/apptainer_${SLURM_JOB_ID}
mkdir -p "$APPTAINER_TMPDIR"
trap 'rm -rf "$APPTAINER_TMPDIR"' EXIT

# Isaac Lab image tag(s) to try, in order (3.0 = Isaac Sim 6.0.1 / Python 3.12)
for TAG in 3.0.0-beta2; do
    SIF="$OUT_DIR/isaaclab_${TAG}.sif"
    echo "=== Trying nvcr.io/nvidia/isaac-lab:${TAG} ==="
    if apptainer pull --force "$SIF" "docker://nvcr.io/nvidia/isaac-lab:${TAG}"; then
        echo "=== SUCCESS: $SIF ==="
        ls -lh "$SIF"
        break
    else
        echo "=== Tag ${TAG} failed ==="
    fi
done

ls "$OUT_DIR"/*.sif   # fails the job if no .sif was created
echo "End at $(date)"
