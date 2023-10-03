#!/bin/bash

#source /opt/intel/oneapi/setvars.sh 
#source /opt/intel/oneAPI/latest/setvars.sh 

rm -rf bin
mkdir -p bin

CXX=mpiicpc
CXXFLAGS="-fsycl -std=c++17 -Wall -O3 -mcmodel=large -xCORE-AVX512 -qopt-zmm-usage=high -fno-sycl-id-queries-fit-in-int -DVARTYPE=double" # -qopt-streaming-stores always
I_MPI_CXX=icpx $CXX $CXXFLAGS flow_dpcpp.cpp -o bin/flow_dpcpp
