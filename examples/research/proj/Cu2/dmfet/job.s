#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH -t 6:00:00
cd $SLURM_SUBMIT_DIR
python  Cu2.py > Cu2.out.10
