#!/bin/bash
salloc -N 4 --ntasks-per-node=4 --gpus-per-node=4 --constraint=gpu --qos=interactive -A deepsrch_g -t 01:00:00 bash -c "
nvidia-smi
module load python
conda activate gigalens
srun -n 4 --gpus-per-task=1 --ntasks-per-node=4 python -u /global/u2/c/chaseg/$1
"