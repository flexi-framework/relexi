#!/bin/bash
#SBATCH --job-name=relexi_HIT_24_DOF
#SBATCH --output=relexi_HIT_24_DOF.out
#SBATCH --error=relexi_HIT_24_DOF.err
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=128 --cpus-per-task=1 --nodes=2
#SBATCH --partition=rome

# Load Modules
#-----------------
#module load XXXX

# Load Python Environment
#-----------------
#source ~/.venv/bin/activate

# Set Environment Variables
#-----------------
# Ensure python writes are directly flushed to stdout
export PYTHONUNBUFFERED=1
# SmartSim and TF Logging
export SMARTSIM_LOG_LEVEL="quiet"
export SMARTSIM_FLAG_TELEMETRY=0
export TF_CPP_MIN_LOG_LEVEL=2

# Start Relexi
relexi prm_train.yaml
