#!/bin/bash -l
#BATCH --job-name=gym_city_cpu_run
#SBATCH --output=logs/alice/gym_city%x_%j.out
#SBATCH --mail-user="<samyhashim2001@gmail.com>"
#SBATCH --mail-type="ALL"
#SBATCH --mem=48G
#SBATCH --time=144:00:00
#SBATCH --partition=cpu-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12

conda activate /home/s3458717/data1/conda/envs/gymcity36_env 

echo "[$SHELL] #### Starting Gym City CPU"
echo "[$SHELL] This is $SLURM_JOB_USER and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] CWD: "$CWD

# Which GPU has been assigned
# echo "[$SHELL] Using GPU: "$CUDA_VISIBLE_DEVICES


# Create a directory for the logs in the working directory
export LOGDIR=$CWD/logs/alice
mkdir -p $LOGDIR
echo "[$SHELL] Log directory "$LOGDIR

# Run the file
echo "[$SHELL] Running gym city on $(hostname)"
python --version
python hyper_opt_run.py --num-frames 10_000_000 --algo a2c  --vec-envs 64 --log False --save True --save-interval 5_000_000 --power-puzzle False
echo "[$SHELL] Terminated. Job done"
