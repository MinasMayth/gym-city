#!/bin/bash
#BATCH --job-name=gym_city_gpu_run
#SBATCH --output=logs/alice/gym_city%x_%j.out
#SBATCH --mail-user="<samyhashim2001@gmail.com>"
#SBATCH --mail-type="ALL"
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2080_ti:1


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
python stable_baseline_run.py --num-frames 2_000_000 --algo dqn --lr 1e-5 --buffer-size 10000 --learning-starts 1000 --batch-size 32 --target-update-interval 1000 --tau 1.0 --map-width 24 --save-interval 500_000 --seed 1 --vec-env 4  --save True

echo "[$SHELL] Terminated. Job done"
