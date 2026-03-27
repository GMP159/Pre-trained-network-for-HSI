#!/bin/bash
#SBATCH --job-name=finetune_mae
#SBATCH --partition=p1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/finetune_mae_%j.out
#SBATCH --error=logs/finetune_mae_%j.err

set -e

echo "Starting MAE Fine-tuning Experiments"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo ""

cd /scratch2/macu69pa/hsi_project
mkdir -p logs

# Activate venv (adjust if needed)
source /scratch2/macu69pa/venv_macu69pa/bin/activate

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

echo "=========================================="
echo "1. MAE PRE-TRAINED (FINE-TUNE FULL MODEL)"
echo "=========================================="
python src2/evaluation/finetune_mae.py --use_pretrained

echo ""
echo "=========================================="
echo "2. FROM SCRATCH (RANDOM INIT, FINE-TUNE FULL MODEL)"
echo "=========================================="
python src2/evaluation/finetune_mae.py

echo ""
echo "=========================================="
echo "ALL MAE EXPERIMENTS COMPLETE"
echo "=========================================="
echo "Time: $(date)"

echo ""
echo "Results summary:"
echo "- MAE pretrained: outputs/finetune_mae_best.pth"
echo "- Random init: outputs/finetune_mae_random_best.pth"