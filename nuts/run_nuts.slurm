#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH --qos=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --job-name=nuts_distributed
#SBATCH --output=results/output.out
#SBATCH --account=deepsrch_g

module load python
conda activate gigalens
srun -n 2 -c 1 --cpu_bind=cores --gpu-bind=none --gpu-bind=single:1 python -u /global/u2/c/chaseg/nuts/run_nuts.py