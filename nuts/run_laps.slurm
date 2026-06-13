#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu
#SBATCH --job-name=laps_256
#SBATCH --output=results/laps_%j.out
#SBATCH --account=deepsrch_g

export JAX_COORDINATOR_ADDR=$(hostname):54321
export JAX_PROCESS_COUNT=4
export JAX_PROCESS_INDEX=$SLURM_PROCID
export NCCL_SOCKET_IFNAME=hsn

module load python
conda activate gigalens

srun -n 4 -c 1 --cpu_bind=cores --gpu-bind=single:1 \
    python -u /global/homes/c/chaseg/nuts/run_laps.py
