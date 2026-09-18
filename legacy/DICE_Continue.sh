module load intel
module load python
module load gcc

export OMP_NUM_THREADS=4
export TMP="/var/scratch/"
export TEMP="/var/scratch/"
export TMPDIR="/var/scratch/"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lgutsev/boost_1_70_0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lgutsev/boost_1_70_0/stage/lib

MPICOMMAND="mpirun -np 20"
HCIPATH="/home/lgutsev/Dice/Dice"

$MPICOMMAND $HCIPATH > output.dat
#python ../test_energy.py 1  1.0e-5
#python ../test_twopdm.py spatialRDM.0.0.txt trusted2RDM.txt 1.e-8


