#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=20
#SBATCH -t 4:00:00

cd $SLURM_SUBMIT_DIR
qchem -nt 20 C24_opt.in C24_opt.out 

