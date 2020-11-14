#!/bin/sh

# define the binary of gromacs
if which gmx_mpi &> /dev/null; then
    GMX_BIN=$(which gmx_mpi)
else which gmx &> /dev/null;
    GMX_BIN=$(which gmx)
fi
echo "Will use the gromacs executable binary at $GMX_BIN"

# input (.gro, .top and .mdp) and output (.tpr)
# MDP_FILE: running parameters
# GRO_FILE: atom positions
# TOP_FILE: topology
MDP_FILE=002_PMF.mdp
GRO_FILE=../p41-abl.new.gro
TOP_FILE=../p41-abl.top

# generate the tpr file
OUTPUT_BASENAME=$(basename $(pwd))
TPR_FILE="$OUTPUT_BASENAME.tpr"
NEW_MDP_FILE="$OUTPUT_BASENAME.mdp"
echo "Making gromacs tpr file ($TPR_FILE)..."
$GMX_BIN grompp -f $MDP_FILE -c $GRO_FILE -p $TOP_FILE -o $TPR_FILE -po $NEW_MDP_FILE

# modify the colvars input file using the PMF minima in previous stages
ARGMIN_001=$(awk -f ./find_min_max.awk ../001_RMSD_bound/output/001_RMSD_bound.out.abf1.czar.pmf)
sed -i "s/001_pmf_argmin/$ARGMIN_001/g" 002_colvars.dat

# run the PMF calculation
mkdir -p output
DEFAULT_OUTPUT_FILENAME="output/$OUTPUT_BASENAME.out"
# $GMX_BIN mdrun -ntmpi 1 -ntomp 4 -nb gpu -pme gpu -gpu_id 0 -s $TPR_FILE -deffnm $DEFAULT_OUTPUT_FILENAME -colvars colvar.dat
echo "You can now run gromacs by the following example:"
echo "$GMX_BIN mdrun -ntmpi 1 -ntomp 4 -nb gpu -pme gpu -gpu_id 0 -s $TPR_FILE -deffnm $DEFAULT_OUTPUT_FILENAME -colvars 002_colvars.dat"
