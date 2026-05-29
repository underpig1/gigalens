#!/bin/bash
#SBATCH -N 8
#SBATCH -C gpu
#SBATCH -G 32
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -A deepsrch_g
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH -o logs/100.out

export OMP_NUM_THREADS=32
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Critical environment variables
export JAX_COORDINATOR_ADDR=$(hostname):54321
#export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn  # Perlmutter's high-speed network [2][7]

srun -c 1 --cpu_bind=cores --gpu-bind=none \
    bash -c "module load python
conda activate gigalens
python ./linus-gigalens/hundredsystems.py 80"