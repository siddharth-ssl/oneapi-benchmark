#!/bin/bash

#source /opt/intel/oneapi/setvars.sh 
#source /opt/intel/oneAPI/latest/setvars.sh 

rm -rf bin
mkdir -p bin

CXX=mpiicpc
CXXFLAGS="-std=c++17 -Wall -O3 -fopenmp -mcmodel=large -xCORE-AVX512 -qopt-zmm-usage=high -DVARTYPE=double"
I_MPI_CXX=icpx $CXX $CXXFLAGS flow_mpi_openmp.cpp -o bin/flow_mpi
