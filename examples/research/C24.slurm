#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH -t 12:00:00

cd $SLURM_SUBMIT_DIR
python C24_dfet_ao.py >& C24_dfet_ao.out 

