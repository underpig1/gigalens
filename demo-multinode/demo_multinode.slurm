#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH --qos=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --job-name=jax_distributed_container_demo
#SBATCH --output=results/jax_distributed_container_demo.out
#SBATCH --account=deepsrch_g

srun -c 1 --cpu_bind=cores --gpu-bind=none \
  shifter --module=gpu,nccl-plugin \
  --image=docker:docker.io/eldenyap/gigalens-multinode:latest \
  python /global/u2/c/chaseg/demo-multinode/demo_multinode.py