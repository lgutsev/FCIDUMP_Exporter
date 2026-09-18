#!/bin/sh
# This is the name of my job
#PBS -N DICE_TEST
#PBS -A loni_alumina02
#PBS -q single
#PBS -l nodes=1:ppn=1
#PBS -l walltime=1:00:00
#PBS -V
#PBS -o par.out
#PBS -e par.err
export WORK_DIR=$PBS_O_WORKDIR
cd $WORK_DIR

export OMP_NUM_THREADS=1

module load intel
module load python

export TMP="/var/scratch/" 
export TEMP="/var/scratch/"
export TMPDIR="/var/scratch/"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lgutsev/boost_1_70_0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lgutsev/boost_1_70_0/stage/lib


./runTests.sh



exit
