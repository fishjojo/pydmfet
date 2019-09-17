#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH -t 2:00:00
cd $SLURM_SUBMIT_DIR
python H_dmfet.py > H_dmfet.out

