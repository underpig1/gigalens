#!/bin/bash
salloc -N 2 --ntasks-per-node=2 --gpus-per-node=2 --constraint=gpu --qos=interactive -A deepsrch_g -t 00:30:00 bash -c "
nvidia-smi
module load python
conda activate gigalens
srun -n 2 --gpu-bind=single:1 python -u /global/u2/c/chaseg/nuts/run_nuts.py
"