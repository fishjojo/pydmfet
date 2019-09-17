#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH -t 1:00:00
cd $SLURM_SUBMIT_DIR
python O2-graphene.py  > O2-graphene.out

