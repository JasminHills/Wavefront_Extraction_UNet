#!/bin/sh
#!/usr/bin/env python
#PBS -N 10psLon
#PBS -l walltime=01:40:00
#PBS -l select=1:ncpus=128:mem=920g:mpiprocs=1:ompthreads=128
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/rds/general/user/jh2619/home/anaconda3/lib
export FI_PROVIDER=tcp

module load anaconda3/personal
cd $HOME/CNN/Old/ML/src/generation/

pwd
python trn-1D.py
