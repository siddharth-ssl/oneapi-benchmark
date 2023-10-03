#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef REAL
  #define REAL float
#endif

__device__ inline size_t
bidx(size_t bx, size_t by, size_t bz, size_t nbx, size_t nby, size_t nbz)
{
  return bx + nbx * (by + nby * bz);
}

__device__ inline size_t
idx(size_t x, size_t y, size_t z, size_t npx, size_t npy, size_t npz)
{
  return x + npx * (y + npy * z);
}

__global__ void 
diffuse_d3q7_device(REAL *a, REAL *anew, size_t npx, size_t npy, size_t npz, size_t np, REAL cfl)
{ 
  size_t b = bidx(blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z);
  size_t bsize = blockDim.x * blockDim.y * blockDim.z;
  REAL* T = &a[bsize * b];
  REAL* Tnew = &anew[bsize * b];

  int x = threadIdx.x;
  int y = threadIdx.y;
  int z = threadIdx.z;

  if (z >= np && z <= npz-(np+1)) {
    if (y >= np && y <= npy-(np+1)) {
      if (x >= np && x <= npx-(np+1)) {
        Tnew[idx(x,y,z,npx,npy,npz)] = (1. - 6. * cfl) * T[idx(x,y,z,npx,npy,npz)]
                                       + cfl * (T[idx(x-1,y,z,npx,npy,npz)] + T[idx(x+1,y,z,npx,npy,npz)]
                                              + T[idx(x,y-1,z,npx,npy,npz)] + T[idx(x,y+1,z,npx,npy,npz)]
                                              + T[idx(x,y,z-1,npx,npy,npz)] + T[idx(x,y,z+1,npx,npy,npz)]);
      }
    }
  }  
}

void
diffuse_blocks_d3q7_device(REAL* T, REAL* Tnew, size_t npx, size_t npy, size_t npz, size_t np, size_t nbx, size_t nby, size_t nbz, REAL cfl)
{
  dim3 blocks(nbx, nby, nbz);
  dim3 threads(npx, npy, npz);
  diffuse_d3q7_device<<<blocks, threads>>>(T, Tnew, npx, npy, npz, np, cfl);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
    fprintf(stderr, "diffuse kernel error: %s\n", cudaGetErrorString(err));
  cudaDeviceSynchronize();
  return;
}

__global__ void 
initialise_device(REAL *a, size_t npx, size_t npy, size_t npz, size_t np, REAL Tbulk, REAL Tbc)
{
  size_t b = bidx(blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z);
  size_t bsize = blockDim.x * blockDim.y * blockDim.z;
  REAL* T = &a[bsize * b];

  int x = threadIdx.x;
  int y = threadIdx.y;
  int z = threadIdx.z;

  if (z >= 0 && z <= npz-1) {
    if (y >= 0 && y <= npy-1) {
      if (x >= 0 && x <= npx-1) {
        T[idx(x,y,z,npx,npy,npz)] = Tbc;
      }
    }
  }
 
  if (z >= np && z <= npz-(np+1)) {
    if (y >= np && y <= npy-(np+1)) {
      if (x >= np && x <= npx-(np+1)) {
        T[idx(x,y,z,npx,npy,npz)] = Tbulk;
      }
    }
  }
}

void
initialise_blocks_device(REAL* T, size_t npx, size_t npy, size_t npz, size_t np, size_t nbx, size_t nby, size_t nbz, REAL Tbulk, REAL Tbc)
{
  dim3 blocks(nbx, nby, nbz);
  dim3 threads(npx, npy, npz);
  initialise_device<<<blocks, threads>>>(T, npx, npy, npz, np, Tbulk, Tbc);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
    fprintf(stderr, "initialise kernel error: %s\n", cudaGetErrorString(err));
  cudaDeviceSynchronize();
  return;
}

void
swap(REAL** a, REAL** b)
{
  REAL* tmp = *a;
  *a = *b;
  *b = tmp;
  return;
}

double
run(size_t nx, size_t ny, size_t nz, size_t nbx, size_t nby, size_t nbz, size_t nt, MPI_Comm mpi_comm)
{
  const size_t np = 1;
  const size_t npx = nx + 2 * np;
  const size_t npy = ny + 2 * np;
  const size_t npz = nz + 2 * np;

  const size_t alloc_bytes = npx * npy * npz * nbx * nby * nbz * sizeof(REAL);
  REAL *T, *Tnew;
  cudaMalloc((void**)&T, alloc_bytes);
  cudaMalloc((void**)&Tnew, alloc_bytes);
  
  initialise_blocks_device(T, npx, npy, npz, np, nbx, nby, nbz, 0., 100.);
  initialise_blocks_device(Tnew, npx, npy, npz, np, nbx, nby, nbz, 0., 100.);

  MPI_Barrier(mpi_comm);
  double tic = MPI_Wtime();
  for (size_t t = 0; t < nt; t++) {
    diffuse_blocks_d3q7_device(T, Tnew, npx, npy, npz, np, nbx, nby, nbz, 0.1);
    swap(&T, &Tnew);
  }
  MPI_Barrier(mpi_comm);
  double toc = MPI_Wtime();

  REAL sample_val = 0;
  cudaMemcpy(&sample_val, &T[np + npx * (np + npy * np)], sizeof(REAL), cudaMemcpyDeviceToHost);

  cudaFree(T);
  cudaFree(Tnew);

  if (sample_val < 0) {
    return -1.0;
  } else {
    return toc - tic;
  }
}

int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int mpi_rank, mpi_size;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_size);

  const size_t nx  = atoi(argv[1]);
  const size_t ny  = atoi(argv[2]);
  const size_t nz  = atoi(argv[3]);
  const size_t nbx = atoi(argv[4]);
  const size_t nby = atoi(argv[5]);
  const size_t nbz = atoi(argv[6]);
  const size_t nt  = atoi(argv[7]);

  double elapsed = run(nx, ny, nz, nbx, nby, nbz, nt, mpi_comm);
  if (0 == mpi_rank) {
    fprintf(stdout, "%d, %lu, %lu, %lu, %lu, %lu, %lu, %lu, %f\n", mpi_size, nx, ny, nz, nbx, nby, nbz, nt, elapsed);
  }

  MPI_Finalize();
  return 0;
}
