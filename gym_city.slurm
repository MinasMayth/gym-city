#!/bin/bash
#BATCH --job-name=gym_city
#SBATCH --output=logs/alice/%x_%j.out
#SBATCH --mail-user="<samyhashim2001@gmail.com>"
#SBATCH --mail-type="ALL"
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# module restore gym_city

echo "[$SHELL] #### Starting Gym City"
echo "[$SHELL] This is $SLURM_JOB_USER and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] CWD: "$CWD

# Which GPU has been assigned
echo "[$SHELL] Using GPU: "$CUDA_VISIBLE_DEVICES


# Create a directory for the logs in the working directory
export LOGDIR=$CWD/logs/alice
mkdir -p $LOGDIR
echo "[$SHELL] Log directory "$LOGDIR

# Run the file
echo "[$SHELL] Running gym city on $(hostname)"
python --version
python stable_baseline_run.py --num-frames 10_000_000 --algo a2c --log False --alpha 0.95
echo "[$SHELL] Terminated. Job done"
