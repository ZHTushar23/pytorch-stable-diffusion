#!/bin/bash 
#SBATCH --job-name=test        # Job name
#SBATCH --mail-user=ztushar1@umbc.edu       # Where to send mail
#SBATCH --mem=40000                       # Job memory request
#SBATCH --gres=gpu:1                     # Number of requested GPU(s) 
#SBATCH --time=72:00:00                   # Time limit days-hrs:min:sec
#SBATCH --constraint=rtx_8000            # Specific hardware constraint
#SBATCH --error=model.err                # Error file name
#SBATCH --output=model.out               # Output file name


export CUDA_LAUNCH_BLOCKING=1
python diffusion2.py