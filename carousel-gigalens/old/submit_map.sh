#!/bin/bash
#SBATCH --job-name=carousel_map
#SBATCH --account=deepsrch_g          # update to your account
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/map_%j.out
#SBATCH --error=logs/map_%j.err

mkdir -p logs

module load python
conda activate gigalens

cd /global/u1/c/chaseg/carousel-gigalens
python run_map.py
