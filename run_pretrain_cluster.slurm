#!/bin/bash
#SBATCH --job-name=hsi_pretrain_tiff
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:7
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err

source /scratch2/macu69pa/venv_macu69pa/bin/activate
cd /scratch2/macu69pa/hsi_project

mkdir -p logs

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

echo "=========================================="
echo "JOB INFO"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

echo "Checking CUDA"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Visible GPUs: {torch.cuda.device_count()}')"

echo "nvidia-smi"
nvidia-smi

NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Detected $NGPUS visible GPUs"

DATA_ROOT="/scratch2/macu69pa/hsi_project/data/patches_64x64_geotiff"
SAVE_DIR="/scratch2/macu69pa/hsi_project/outputs/pretrain_enhanced"

BATCH_SIZE=8
EPOCHS=60
LR=1e-4
NUM_WORKERS=4
DEPTH=4
PATCH_H=2
PATCH_W=2
PATCH_C=16
MODE="contrastive"

if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Data directory not found: $DATA_ROOT"
    exit 1
fi

TIFF_COUNT=$(find "$DATA_ROOT" -name '*.tif*' | wc -l)
echo "Data root: $DATA_ROOT"
echo "Found $TIFF_COUNT TIFF files"
echo "Mode: $MODE"
echo "Patch size: ${PATCH_H}x${PATCH_W} spatial, ${PATCH_C} spectral"
echo "Batch size: $BATCH_SIZE per GPU"
echo "Effective batch size: $((BATCH_SIZE * NGPUS))"
echo "=========================================="

torchrun \
    --nproc_per_node=$NGPUS \
    --master_port=29500 \
    run_pretrain_enhanced.py \
    --data_root "$DATA_ROOT" \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --epochs $EPOCHS \
    --lr $LR \
    --depth $DEPTH \
    --patch_h $PATCH_H \
    --patch_w $PATCH_W \
    --patch_c $PATCH_C \
    --mode $MODE \
    --save_dir "$SAVE_DIR"

RESULT=$?

echo "=========================================="
echo "Exit code: $RESULT"
echo "End time: $(date)"
echo "=========================================="

exit $RESULT
