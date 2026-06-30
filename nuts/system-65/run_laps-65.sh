#!/bin/bash
salloc -N 4 --ntasks-per-node=4 --gpus-per-node=4 --constraint=gpu --qos=interactive -A deepsrch_g -t 00:30:00 --mem=0 bash -c "
nvidia-smi
module load python
conda activate gigalens
export JAX_COORDINATOR_ADDR=$(hostname):54321
export JAX_PROCESS_COUNT=16
export JAX_PROCESS_INDEX=\$SLURM_PROCID
export NCCL_SOCKET_IFNAME=hsn
srun -n 16 --ntasks-per-node=4 --gpus-per-node=4 python -u /global/homes/c/chaseg/nuts/system-65/laps-65.py
"
