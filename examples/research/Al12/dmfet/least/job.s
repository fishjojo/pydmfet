#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH -t 12:00:00
cd $SLURM_SUBMIT_DIR
python  Al12.py > Al12.out.10
