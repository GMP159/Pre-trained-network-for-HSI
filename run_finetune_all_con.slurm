#!/bin/bash
#SBATCH --job-name=finetune_contrastive
#SBATCH --partition=p1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/finetune_contrastive_%j.out
#SBATCH --error=logs/finetune_contrastive_%j.err

set -e

echo "Starting Contrastive Fine-tuning Experiments"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo ""

cd /scratch2/macu69pa/hsi_project
mkdir -p logs

# Activate venv (adjust if your path is different)
source /scratch2/macu69pa/venv_macu69pa/bin/activate

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

echo "=========================================="
echo "1. CONTRASTIVE PRE-TRAINED (FINE-TUNE FULL MODEL)"
echo "=========================================="
python src2/evaluation/finetune_contrastive.py --use_pretrained

echo ""
echo "=========================================="
echo "2. FROM SCRATCH (RANDOM INIT, FINE-TUNE FULL MODEL)"
echo "=========================================="
python src2/evaluation/finetune_contrastive.py

echo ""
echo "=========================================="
echo "ALL CONTRASTIVE EXPERIMENTS COMPLETE"
echo "=========================================="
echo "Time: $(date)"

echo ""
echo "Results summary:"
echo "- Contrastive pretrained: outputs/finetune_contrastive_best.pth"
echo "- Random init:            outputs/finetune_contrastive_random_best.pth"
