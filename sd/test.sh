#!/bin/bash 
#SBATCH --job-name=scratch        # Job name
#SBATCH --mail-user=ztushar1@umbc.edu       # Where to send mail
#SBATCH --mem=20000                       # Job memory request
#SBATCH --gres=gpu:1                     # Number of requested GPU(s) 
#SBATCH --time=72:00:00                   # Time limit days-hrs:min:sec
#SBATCH --constraint=rtx_6000            # Specific hardware constraint
#SBATCH --error=results.err                # Error file name
#SBATCH --output=results.out               # Output file name


export CUDA_LAUNCH_BLOCKING=1
python demo.py