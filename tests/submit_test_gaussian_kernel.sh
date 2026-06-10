#!/bin/bash
#SBATCH --clusters=gmerlin7
#SBATCH --partition=a100-general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=triton_gauss_test
#SBATCH --output=/data/project/cls/standfuss/torchsx/tests/logs/%x_%j.out
set -euo pipefail

source /etc/profile.d/pmodules.sh
module load Python/3.12.9

HERE=/data/project/cls/standfuss/torchsx
mkdir -p "$HERE/tests/logs"
cd "$HERE"

exec /data/user/seidel_h/.env_torchsx/bin/python -u tests/test_gaussian_kernel.py
