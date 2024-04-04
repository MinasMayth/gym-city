#!/bin/bash
#BATCH --job-name=gym_city_gpu_run
#SBATCH --output=logs/alice/gym_city%x_%j.out
#SBATCH --mail-user="<samyhashim2001@gmail.com>"
#SBATCH --mail-type="ALL"
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-medium
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
python stable_baseline_run.py --num-frames 15_000_000 --algo a2c  --alpha 0.99 --value-loss-coef 0.5 --max-grad-norm 0.5 --entropy-coef 0.01 --lr 9e-4 --eps 1e-5 --gae 0.98 --gamma 0.98 --num-steps 11 --map-width 96
echo "[$SHELL] Terminated. Job done"
