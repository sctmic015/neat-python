#!/bin/bash

#PBS -l select=3:ncpus=24:mpiprocs=24:nodetype=haswell_reg
#PBS -P CSCI1142
#PBS -q normal
#PBS -l walltime=0:15:00
#PBS -o /mnt/lustre/users/mscott/neat-python/map.out
#PBS -e /mnt/lustre/users/mscott/neat-python/map.err
#PBS -m abe
#PBS -M sctmic015@myuct.ac.za
#PBS -N mapTestPar

ulimit -s 10240

module purge
module load chpc/python/3.7.0

cd $PBS_O_WORKDIR
nproc=`cat $PBS_NODEFILE | wc -l`
mpirun -np $nproc python3 -m mpi4py.futures mapElitesNEAT2.py 20000 5