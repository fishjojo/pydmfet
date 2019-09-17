#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH -t 2:00:00
cd $SLURM_SUBMIT_DIR
python  water2.py > water2.out

