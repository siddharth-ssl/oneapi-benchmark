#!/bin/bash

source /opt/intel/oneapi/setvars.sh 
#source /opt/intel/oneAPI/latest/setvars.sh 

rm -rf bin
mkdir -p bin

#CXX=mpiicpc
#CXXFLAGS="-std=c++17 -Wall -O3 -mcmodel=large -xCORE-AVX512 -qopt-zmm-usage=high -DVARTYPE=double"
#$CXX $CXXFLAGS flow_mpi.cpp -o bin/flow_mpi

CXX=mpiicpc
CXXFLAGS="-fsycl -std=c++17 -Wall -O3 -mcmodel=large -xCORE-AVX512 -qopt-zmm-usage=high -fno-sycl-id-queries-fit-in-int -DVARTYPE=double" # -qopt-streaming-stores always
I_MPI_CXX=icpx $CXX $CXXFLAGS flow_dpcpp_block.cpp -o bin/flow_dpcpp

CXX=mpiicpc
CXXFLAGS="-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 -std=c++17 -Wall -O3 -mcmodel=large -xCORE-AVX512 -qopt-zmm-usage=high -fno-sycl-id-queries-fit-in-int -DVARTYPE=double" # -qopt-streaming-stores always
I_MPI_CXX=clang++ $CXX $CXXFLAGS flow_dpcpp_block.cpp -o bin/flow_dpcpp_nv

#CXX=mpiicpc
#CXXFLAGS="-fsycl -std=c++17 -Wall -O3 -mcmodel=large -xCORE-AVX512 -qopt-zmm-usage=high -fno-sycl-id-queries-fit-in-int -DVARTYPE=double" # -qopt-streaming-stores always
#I_MPI_CXX=dpcpp $CXX $CXXFLAGS flow_dpcpp_block.cpp -o bin/flow_dpcpp_block
