#!/bin/bash
salloc -N 8 --ntasks-per-node=32 --gpus-per-node=32 --constraint=gpu --qos=debug -A deepsrch_g -t 00:30:00 bash -c "
nvidia-smi
module load python
conda activate gigalens
export JAX_COORDINATOR_ADDR=$(hostname):54321
export JAX_PROCESS_COUNT=32
export JAX_PROCESS_INDEX=$SLURM_PROCID
export NCCL_SOCKET_IFNAME=hsn
srun -n 32 --ntasks-per-node=32 --gpus-per-node=32 python -u /global/u2/c/chaseg/nuts/run_nuts.py
"