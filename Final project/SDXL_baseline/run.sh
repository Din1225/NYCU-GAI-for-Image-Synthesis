#!/bin/bash
#SBATCH --job-name=dif3   # 作業名稱
#SBATCH --output=dif3_log.txt  # 標準輸出文件
#SBATCH --error=dif3_error_module.txt        # 標準錯誤文件
#SBATCH --ntasks=1                      # 任務數量(執行次數)
#SBATCH --time=40:00:00                 # 設定運行時間
#SBATCH -p defq                        # 設定使用分區：測試程式碼使用 trialq
#SBATCH --gres=gpu:1

#==========================
# Load modules
#==========================

module purge
module load anaconda
module list
#==========================
# Anaconda Environment
#==========================
eval "$(conda shell.bash hook)"
conda deactivate
conda activate genai
#==========================
# Execute My Program
#==========================
# 執行 Python 程序
export PYTHONUNBUFFERED=1
cd /home/a00164/oscar50513.ii13/mg/genai/SDXL_baseline;
python diffusion.py;