#include <mpi.h>
#include <chrono>
#include <CL/sycl.hpp>
#include <dpc_common.hpp>
#include <tuple>
#include <cstdlib>
#include <iostream>

#ifndef VARTYPE
  #define VARTYPE double
#endif

#define NG 4UL
#define NM 8UL

#define BETA2  1.
#define ITHETA 3.
#define W000   (64./216.)
#define W100   (16./216.)
#define W110   ( 4./216.)
#define W111   ( 1./216.)

inline size_t
idx(size_t m, size_t x, size_t y, size_t z, size_t g, size_t npx, size_t npy, size_t npz)
{
  return m + NM * (x + npx * (y + npy * (z + npz * g)));
}

void
fill_moments(VARTYPE* f, VARTYPE* rho1, VARTYPE* ux1, VARTYPE* uy1, VARTYPE* uz1)
{
  VARTYPE rho = 0;
  VARTYPE ux  = 0;
  VARTYPE uy  = 0;
  VARTYPE uz  = 0;
  VARTYPE fval;

  fval = f[0 + NM * 0]; rho += fval; ux += +1*fval; uy += +1*fval; uz += -1*fval;
  fval = f[1 + NM * 0]; rho += fval; ux += +0*fval; uy += +1*fval; uz += -1*fval;
  fval = f[2 + NM * 0]; rho += fval; ux += -1*fval; uy += +1*fval; uz += -1*fval;
  fval = f[3 + NM * 0]; rho += fval; ux += +1*fval; uy += +0*fval; uz += -1*fval;
  fval = f[4 + NM * 0]; rho += fval; ux += +0*fval; uy += +0*fval; uz += -1*fval;
  fval = f[5 + NM * 0]; rho += fval; ux += -1*fval; uy += +0*fval; uz += -1*fval;
  fval = f[6 + NM * 0]; rho += fval; ux += +1*fval; uy += -1*fval; uz += -1*fval;

  fval = f[0 + NM * 1]; rho += fval; ux += -1*fval; uy += -1*fval; uz += +1*fval;
  fval = f[1 + NM * 1]; rho += fval; ux += +0*fval; uy += -1*fval; uz += +1*fval;
  fval = f[2 + NM * 1]; rho += fval; ux += +1*fval; uy += -1*fval; uz += +1*fval;
  fval = f[3 + NM * 1]; rho += fval; ux += -1*fval; uy += +0*fval; uz += +1*fval;
  fval = f[4 + NM * 1]; rho += fval; ux += +0*fval; uy += +0*fval; uz += +1*fval;
  fval = f[5 + NM * 1]; rho += fval; ux += +1*fval; uy += +0*fval; uz += +1*fval;
  fval = f[6 + NM * 1]; rho += fval; ux += -1*fval; uy += +1*fval; uz += +1*fval;

  fval = f[0 + NM * 2]; rho += fval; ux += +0*fval; uy += -1*fval; uz += -1*fval;
  fval = f[1 + NM * 2]; rho += fval; ux += -1*fval; uy += -1*fval; uz += -1*fval;
  fval = f[2 + NM * 2]; rho += fval; ux += +0*fval; uy += +0*fval; uz += +0*fval;
  fval = f[3 + NM * 2]; rho += fval; ux += -1*fval; uy += +0*fval; uz += +0*fval;
  fval = f[4 + NM * 2]; rho += fval; ux += +1*fval; uy += -1*fval; uz += +0*fval;
  fval = f[5 + NM * 2]; rho += fval; ux += +0*fval; uy += -1*fval; uz += +0*fval;
  fval = f[6 + NM * 2]; rho += fval; ux += -1*fval; uy += -1*fval; uz += +0*fval;

  fval = f[0 + NM * 3]; rho += fval; ux += +0*fval; uy += +1*fval; uz += +1*fval;
  fval = f[1 + NM * 3]; rho += fval; ux += +1*fval; uy += +1*fval; uz += +1*fval;
  fval = f[3 + NM * 3]; rho += fval; ux += +1*fval; uy += +0*fval; uz += +0*fval;
  fval = f[4 + NM * 3]; rho += fval; ux += -1*fval; uy += +1*fval; uz += +0*fval;
  fval = f[5 + NM * 3]; rho += fval; ux += +0*fval; uy += +1*fval; uz += +0*fval;
  fval = f[6 + NM * 3]; rho += fval; ux += +1*fval; uy += +1*fval; uz += +0*fval;

  *rho1 = rho;
  *ux1 = ux / rho;
  *uy1 = uy / rho;
  *uz1 = uz / rho;
  
  return;
}

void
fill_feq(VARTYPE rho, VARTYPE ux, VARTYPE uy, VARTYPE uz, VARTYPE* f)
{
  const VARTYPE udotu = (ux*ux + uy*uy + uz*uz) * ITHETA;
  VARTYPE cdotu;

  // g0
  cdotu = (+1*ux + +1*uy + -1*uz) * ITHETA;
  f[0 + NM * 0] = rho * W111 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+0*ux + +1*uy + -1*uz) * ITHETA;
  f[1 + NM * 0] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (-1*ux + +1*uy + -1*uz) * ITHETA;
  f[2 + NM * 0] = rho * W111 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+1*ux + +0*uy + -1*uz) * ITHETA;
  f[3 + NM * 0] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+0*ux + +0*uy + -1*uz) * ITHETA;
  f[4 + NM * 0] = rho * W100 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (-1*ux + +0*uy + -1*uz) * ITHETA;
  f[5 + NM * 0] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+1*ux + -1*uy + -1*uz) * ITHETA;
  f[6 + NM * 0] = rho * W111 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  // g1
  cdotu = (-1*ux + -1*uy + +1*uz) * ITHETA;
  f[0 + NM * 1] = rho * W111 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+0*ux + -1*uy + +1*uz) * ITHETA;
  f[1 + NM * 1] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+1*ux + -1*uy + +1*uz) * ITHETA;
  f[2 + NM * 1] = rho * W111 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (-1*ux + +0*uy + +1*uz) * ITHETA;
  f[3 + NM * 1] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+0*ux + +0*uy + +1*uz) * ITHETA;
  f[4 + NM * 1] = rho * W100 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+1*ux + +0*uy + +1*uz) * ITHETA;
  f[5 + NM * 1] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (-1*ux + +1*uy + +1*uz) * ITHETA;
  f[6 + NM * 1] = rho * W111 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  // g2
  cdotu = (+0*ux + -1*uy + -1*uz) * ITHETA;
  f[0 + NM * 2] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (-1*ux + -1*uy + -1*uz) * ITHETA;
  f[1 + NM * 2] = rho * W111 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+0*ux + +0*uy + +0*uz) * ITHETA;
  f[2 + NM * 2] = rho * W000 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (-1*ux + +0*uy + +0*uz) * ITHETA;
  f[3 + NM * 2] = rho * W100 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+1*ux + -1*uy + +0*uz) * ITHETA;
  f[4 + NM * 2] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+0*ux + -1*uy + +0*uz) * ITHETA;
  f[5 + NM * 2] = rho * W100 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (-1*ux + -1*uy + +0*uz) * ITHETA;
  f[6 + NM * 2] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  // g3
  cdotu = (+0*ux + +1*uy + +1*uz) * ITHETA;
  f[0 + NM * 3] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+1*ux + +1*uy + +1*uz) * ITHETA;
  f[1 + NM * 3] = rho * W111 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+1*ux + +0*uy + +0*uz) * ITHETA;
  f[3 + NM * 3] = rho * W100 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (-1*ux + +1*uy + +0*uz) * ITHETA;
  f[4 + NM * 3] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+0*ux + +1*uy + +0*uz) * ITHETA;
  f[5 + NM * 3] = rho * W100 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+1*ux + +1*uy + +0*uz) * ITHETA;
  f[6 + NM * 3] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  return;
}

std::tuple<float, float, VARTYPE, VARTYPE, VARTYPE, VARTYPE>
run(size_t nx, size_t ny, size_t nz, size_t nbx, size_t nby, size_t nbz, size_t nt, MPI_Comm mpi_comm, sycl::device& d)
{
  const size_t np = 1;
  const size_t npx = nx + 2 * np;
  const size_t npy = ny + 2 * np;
  const size_t npz = nz + 2 * np;

  sycl::property_list properties{ sycl::property::queue::in_order() };
  sycl::queue q(d, dpc_common::exception_handler, properties);

  const size_t alloc_bytes = NM * (npx * npy * npz) * NG * (nbx * nby * nbz) * sizeof(VARTYPE);
  VARTYPE* T = (VARTYPE*)sycl::malloc_device(alloc_bytes, q);

  sycl::range<3> threads = { nbz, nby, nbx };
  sycl::range<3> wg = { 1, 1, 1 };
  const size_t bsize = NM * (npx * npy * npz) * NG;

  // initialise
  auto initialise_blocks = [&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<3>(threads, wg), [=](sycl::nd_item<3> it) {
      const size_t bx = it.get_group(2);
      const size_t by = it.get_group(1);
      const size_t bz = it.get_group(0);
      const size_t bidx  = (bx + nbx * (by + nby * bz));
      VARTYPE* b = &T[bsize * bidx]; 

      VARTYPE feq[NM * NG] = {0};
      VARTYPE rho = 1., ux = 0., uy = 0., uz = 0.;
      for (size_t z = 0; z <= npz-1; z++) {
        for (size_t y = 0; y <= npy-1; y++) {
          for (size_t x = 0; x <= npx-1; x++) {
            fill_feq(rho, ux, uy, uz, feq);
            for (size_t g = 0; g < NG; g++) {
              for (size_t m = 0; m < NM; m++) {
                b[idx(m,x,y,z,g,npx,npy,npz)] = feq[m + NM * g];
              }
            }
          }
        }
      }
    });
  };

  try {
    q.submit(initialise_blocks);
    q.wait();
  } catch (sycl::exception const& ex) {
    std::cerr << "dpcpp error: " << ex.what() << std::endl;
  }

  // collide
  auto collide_blocks_d3q27 = [&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<3>(threads, wg), [=](sycl::nd_item<3> it) {
      const size_t bx = it.get_group(2);
      const size_t by = it.get_group(1);
      const size_t bz = it.get_group(0);
      const size_t bidx  = (bx + nbx * (by + nby * bz));
      VARTYPE* b = &T[bsize * bidx]; 

      for (size_t z = np; z <= npz-(np+1); z++) {
        for (size_t y = np; y <= npy-(np+1); y++) {
          for (size_t x = np; x <= npx-(np+1); x++) {
            VARTYPE feq[NM * NG] = {0};
            VARTYPE rho, ux, uy, uz;
            for (size_t g = 0; g < NG; g++) {
              for (size_t m = 0; m < NM; m++) {
                feq[m + NM * g] = b[idx(m,x,y,z,g,npx,npy,npz)];
              }
            }
            fill_moments(feq, &rho, &ux, &uy, &uz);
            fill_feq(rho, ux, uy, uz, feq);
            for (size_t g = 0; g < NG; g++) {
              for (size_t m = 0; m < NM; m++) {
                b[idx(m,x,y,z,g,npx,npy,npz)] = (1. - BETA2) * b[idx(m,x,y,z,g,npx,npy,npz)]
                                                + BETA2 * feq[m + NM * g];
              }
            }
          }
        }
      }
    });
  };

  // advect
  auto advect_blocks_d3q27 = [&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<3>(threads, wg), [=](sycl::nd_item<3> it) {
      const size_t bx = it.get_group(2);
      const size_t by = it.get_group(1);
      const size_t bz = it.get_group(0);
      const size_t bidx  = (bx + nbx * (by + nby * bz));
      VARTYPE* b = &T[bsize * bidx]; 

      // Communicate
      size_t plus_ngb_b, neg_ngb_b, b_ngb_id; 

      // X Comm
      if( bx == 0 ) // first block setting periodic in -ve dir
        neg_ngb_b = nbx - 1;
      else
        neg_ngb_b = bx - 1;

      if( bx == nbx - 1 ) // last block setting periodic in +ve dir
        plus_ngb_b = 0;
      else
        plus_ngb_b = bx + 1;

      b_ngb_id = (neg_ngb_b + nbx * (by + nby * bz));
      VARTYPE* b_neg_ngb = &T[bsize * b_ngb_id];
      for (size_t z = 0; z < npz; z++) {
        for (size_t y = 0; y < npy; y++) {

          for (size_t g = 0; g < NG; g++) {
            for (size_t m = 0; m < NM; m++) {
              b[idx(m,0,y,z,g,npx,npy,npz)] = b_neg_ngb[idx(m,npx-(np+1),y,z,g,npx,npy,npz)];
            }
          }

        }
      }

      b_ngb_id = (plus_ngb_b + nbx * (by + nby * bz));
      VARTYPE* b_plus_ngb = &T[bsize * b_ngb_id];
      for (size_t z = 0; z < npz; z++) {
        for (size_t y = 0; y < npy; y++) {

          for (size_t g = 0; g < NG; g++) {
            for (size_t m = 0; m < NM; m++) {
              b[idx(m,npx-1,y,z,g,npx,npy,npz)] = b_plus_ngb[idx(m,np,y,z,g,npx,npy,npz)];
            }
          }

        }
      }
      // Y Comm
      if( by == 0 ) // first block setting periodic in -ve dir
        neg_ngb_b = nby - 1;
      else
        neg_ngb_b = by - 1;

      if( by == nby - 1 ) // last block setting periodic in +ve dir
        plus_ngb_b = 0;
      else
        plus_ngb_b = by + 1;

      b_ngb_id = (bx + nbx * (neg_ngb_b + nby * bz));
      b_neg_ngb = &T[bsize * b_ngb_id];

      for (size_t z = 0; z < npz; z++) {
        for (size_t x = 0; x < npx; x++) {

          for (size_t g = 0; g < NG; g++) {
            for (size_t m = 0; m < NM; m++) {
              b[idx(m,x,0,z,g,npx,npy,npz)] = b_neg_ngb[idx(m,x,npy-(np+1),z,g,npx,npy,npz)];
            }
          }

        }
      }

      b_ngb_id = (bx + nbx * (plus_ngb_b + nby * bz));
      b_plus_ngb = &T[bsize * b_ngb_id];

      for (size_t z = 0; z < npz; z++) {
        for (size_t x = 0; x < npx; x++) {

            for (size_t g = 0; g < NG; g++) {
              for (size_t m = 0; m < NM; m++) {
                b[idx(m,x,npy-1,z,g,npx,npy,npz)] = b_plus_ngb[idx(m,x,np,z,g,npx,npy,npz)];
              }
            }

        }
      }
      // Z comm
      if( bz == 0 ) // first block setting periodic in -ve dir
        neg_ngb_b = nbz - 1;
      else
        neg_ngb_b = bz - 1;

      if( bz == nbz - 1 ) // last block setting periodic in +ve dir
        plus_ngb_b = 0;
      else
        plus_ngb_b = bz + 1;

      b_ngb_id = (bx + nbx * (by + nby * neg_ngb_b));
      b_neg_ngb = &T[bsize * b_ngb_id];

      for (size_t y = 0; y < npy; y++) {
        for (size_t x = 0; x < npx; x++) {

          for (size_t g = 0; g < NG; g++) {
            for (size_t m = 0; m < NM; m++) {
              b[idx(m,x,y,0,g,npx,npy,npz)] = b_neg_ngb[idx(m,x,y,npz-(np+1),g,npx,npy,npz)];
            }
          }

        }
      }

      b_ngb_id = (bx + nbx * (by + nby * plus_ngb_b));
      b_plus_ngb = &T[bsize * b_ngb_id];

      for (size_t y = 0; y < npy; y++) {
        for (size_t x = 0; x < npx; x++) {

          for (size_t g = 0; g < NG; g++) {
            for (size_t m = 0; m < NM; m++) {
              b[idx(m,x,y,npz-1,g,npx,npy,npz)] = b_plus_ngb[idx(m,x,y,np,g,npx,npy,npz)];
            }
          }

        }
      }

      for (size_t z = np; z <= npz-(np+1); z++) {
        for (size_t y = np; y <= npy-(np+1); y++) {
          for (size_t x = np; x <= npx-(np+1); x++) {
            b[idx(0,x,y,z,0,npx,npy,npz)] = b[idx(0,x-1,y-1,z+1,0,npx,npy,npz)];
            b[idx(1,x,y,z,0,npx,npy,npz)] = b[idx(1,x+0,y-1,z+1,0,npx,npy,npz)];
            b[idx(2,x,y,z,0,npx,npy,npz)] = b[idx(2,x+1,y-1,z+1,0,npx,npy,npz)];
            b[idx(3,x,y,z,0,npx,npy,npz)] = b[idx(3,x-1,y+0,z+1,0,npx,npy,npz)];
            b[idx(4,x,y,z,0,npx,npy,npz)] = b[idx(4,x+0,y+0,z+1,0,npx,npy,npz)];
            b[idx(5,x,y,z,0,npx,npy,npz)] = b[idx(5,x+1,y+0,z+1,0,npx,npy,npz)];
            b[idx(6,x,y,z,0,npx,npy,npz)] = b[idx(6,x-1,y+1,z+1,0,npx,npy,npz)];
          }
        }
      }
      for (size_t z = npz-(np+1); z >= np; z--) {
        for (size_t y = npy-(np+1); y >= np; y--) {
          for (size_t x = npx-(np+1); x >= np; x--) {
            b[idx(0,x,y,z,1,npx,npy,npz)] = b[idx(0,x+1,y+1,z-1,1,npx,npy,npz)];
            b[idx(1,x,y,z,1,npx,npy,npz)] = b[idx(1,x+0,y+1,z-1,1,npx,npy,npz)];
            b[idx(2,x,y,z,1,npx,npy,npz)] = b[idx(2,x-1,y+1,z-1,1,npx,npy,npz)];
            b[idx(3,x,y,z,1,npx,npy,npz)] = b[idx(3,x+1,y+0,z-1,1,npx,npy,npz)];
            b[idx(4,x,y,z,1,npx,npy,npz)] = b[idx(4,x+0,y+0,z-1,1,npx,npy,npz)];
            b[idx(5,x,y,z,1,npx,npy,npz)] = b[idx(5,x-1,y+0,z-1,1,npx,npy,npz)];
            b[idx(6,x,y,z,1,npx,npy,npz)] = b[idx(6,x+1,y-1,z-1,1,npx,npy,npz)];
          }
        }
      }
      for (size_t z = np; z <= npz-(np+1); z++) {
        for (size_t y = np; y <= npy-(np+1); y++) {
          for (size_t x = np; x <= npx-(np+1); x++) {
            b[idx(0,x,y,z,2,npx,npy,npz)] = b[idx(0,x+0,y+1,z+1,2,npx,npy,npz)];
            b[idx(1,x,y,z,2,npx,npy,npz)] = b[idx(1,x+1,y+1,z+1,2,npx,npy,npz)];
            b[idx(2,x,y,z,2,npx,npy,npz)] = b[idx(2,x+0,y+0,z+0,2,npx,npy,npz)];
            b[idx(3,x,y,z,2,npx,npy,npz)] = b[idx(3,x+1,y+0,z+0,2,npx,npy,npz)];
            b[idx(4,x,y,z,2,npx,npy,npz)] = b[idx(4,x-1,y+1,z+0,2,npx,npy,npz)];
            b[idx(5,x,y,z,2,npx,npy,npz)] = b[idx(5,x+0,y+1,z+0,2,npx,npy,npz)];
            b[idx(6,x,y,z,2,npx,npy,npz)] = b[idx(6,x+1,y+1,z+0,2,npx,npy,npz)];
          }
        }
      }
      for (size_t z = npz-(np+1); z >= np; z--) {
        for (size_t y = npy-(np+1); y >= np; y--) {
          for (size_t x = npx-(np+1); x >= np; x--) {
            b[idx(0,x,y,z,3,npx,npy,npz)] = b[idx(0,x+0,y-1,z-1,3,npx,npy,npz)];
            b[idx(1,x,y,z,3,npx,npy,npz)] = b[idx(1,x-1,y-1,z-1,3,npx,npy,npz)];
            b[idx(2,x,y,z,3,npx,npy,npz)] = b[idx(2,x+0,y+0,z+0,3,npx,npy,npz)];
            b[idx(3,x,y,z,3,npx,npy,npz)] = b[idx(3,x-1,y+0,z+0,3,npx,npy,npz)];
            b[idx(4,x,y,z,3,npx,npy,npz)] = b[idx(4,x+1,y-1,z+0,3,npx,npy,npz)];
            b[idx(5,x,y,z,3,npx,npy,npz)] = b[idx(5,x+0,y-1,z+0,3,npx,npy,npz)];
            b[idx(6,x,y,z,3,npx,npy,npz)] = b[idx(6,x-1,y-1,z+0,3,npx,npy,npz)];
          }
        }
      }
    });
  };

  float collide_time = 0, advect_time = 0;
  try {
    for (size_t t = 0; t < nt; t++) {
      MPI_Barrier(mpi_comm);
      q.wait();
      auto tic1 = std::chrono::high_resolution_clock::now();
      q.submit(collide_blocks_d3q27);
      q.wait();
      MPI_Barrier(mpi_comm);
      q.wait();
      auto tic2 = std::chrono::high_resolution_clock::now();
      q.submit(advect_blocks_d3q27);
      q.wait();
      MPI_Barrier(mpi_comm);
      auto tic3 = std::chrono::high_resolution_clock::now();
      auto elapsed_time = (std::chrono::duration<double, std::nano>(
                   tic2 - tic1).count())*1E-9;
      collide_time += elapsed_time;

      elapsed_time = (std::chrono::duration<double, std::nano>(
                         tic3 - tic2).count())*1E-9;

      advect_time += elapsed_time;
    }
  } catch (sycl::exception const& ex) {
    std::cerr << "dpcpp error: " << ex.what() << std::endl;
  }

  VARTYPE sample_val;
  q.memcpy(&sample_val, &T[np + npx * (np + npy * np)], sizeof(VARTYPE));
  VARTYPE feq[NM * NG];
  for (size_t g = 0; g < NG; g++) {
    for (size_t m = 0; m < NM; m++) {
      q.memcpy(&feq[m + NM * g], &T[idx(m,np,np,np,g,npx,npy,npz)], sizeof(VARTYPE));
    }
  }
  VARTYPE rho, ux, uy, uz;
  fill_moments(feq, &rho, &ux, &uy, &uz);

  sycl::free(T, q);

  return std::make_tuple(collide_time, advect_time, rho, ux, uy, uz);
}

int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int mpi_rank, mpi_size;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_size);

  sycl::default_selector d_selector;
  sycl::device d = sycl::device(d_selector);

  const size_t nx  = atoi(argv[1]);
  const size_t ny  = atoi(argv[2]);
  const size_t nz  = atoi(argv[3]);
  const size_t nbx = atoi(argv[4]);
  const size_t nby = atoi(argv[5]);
  const size_t nbz = atoi(argv[6]);
  const size_t nt  = atoi(argv[7]);

  auto res = run(nx, ny, nz, nbx, nby, nbz, nt, mpi_comm, d);
  if (0 == mpi_rank) {
    std::cout << nx << ", ";
    std::cout << ny << ", ";
    std::cout << nz << ", ";
    std::cout << nbx << ", ";
    std::cout << nby << ", ";
    std::cout << nbz << ", ";
    std::cout << nt << ", ";
    std::cout << mpi_size << ", ";
    std::cout << std::get<0>(res) << ", "; // collide time
    std::cout << std::get<1>(res) << ", "; // advect time
    std::cout << std::get<2>(res) << ", "; // rho
    std::cout << std::get<3>(res) << ", "; // ux
    std::cout << std::get<4>(res) << ", "; // uy
    std::cout << std::get<5>(res) << ", "; // uz
    std::cout << std::endl;
  }

  MPI_Finalize();
  return 0;
}