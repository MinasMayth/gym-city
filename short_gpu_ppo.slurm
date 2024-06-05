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
python stable_baseline_run.py --num-frames 2_000_000 --algo ppo --value-loss-coef 0.5 --max-grad-norm 0.5 --entropy-coef 0.01 --lr 1e-3 --gae 0.95 --gamma 0.95 --num-steps 1024 --map-width 24 --save-interval 500_000 --num-mini-batch 128 --ppo-epoch 10 --clip-param 0.2 --seed 1 --vec-envs 4 --save True

echo "[$SHELL] Terminated. Job done"
