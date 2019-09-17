#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH -t 3:00:00
cd $SLURM_SUBMIT_DIR
python embed.py > embed.out.2

