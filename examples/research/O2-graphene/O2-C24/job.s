#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH -t 6:00:00
cd $SLURM_SUBMIT_DIR
python O2-C24.py   > O2-C24.out.1

