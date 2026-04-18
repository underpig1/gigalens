#!/bin/bash
salloc -N 4 --ntasks-per-node=4 --gpus-per-node=4 --constraint=gpu --qos=interactive -A deepsrch_g -t 00:30:00 bash -c "
nvidia-smi
module load python
conda activate gigalens
srun -n 4 --gpu-bind=single:1 --ntasks-per-node=4 --gpus-per-node=4 python -u /global/u2/c/chaseg/nuts/run_nuts.py
"