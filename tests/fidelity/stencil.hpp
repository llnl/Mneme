// Adapted from HeCBench; see NOTICE and licenses/.
#pragma once

// 2D block size
#define BSIZE 16
// Tile size in the x direction
#define XTILE 20

typedef double Real;

__global__ void
stencil3d(const Real *__restrict__ d_psi, Real *__restrict__ d_npsi,
          const Real *__restrict__ d_sigmaX, const Real *__restrict__ d_sigmaY,
          const Real *__restrict__ d_sigmaZ, int nx, int ny, int nz) {
  // Three x planes form a sliding stencil window; the fourth shares face fluxes
  // between adjacent threads. Within each plane, z varies fastest.
  __shared__ Real sm_psi[4][BSIZE][BSIZE];

#define V0(y, z) sm_psi[pii][y][z]
#define V1(y, z) sm_psi[cii][y][z]
#define V2(y, z) sm_psi[nii][y][z]

#define sigmaX(x, y, z, dir) d_sigmaX[z + nz * (y + ny * (x + nx * dir))]
#define sigmaY(x, y, z, dir) d_sigmaY[z + nz * (y + ny * (x + nx * dir))]
#define sigmaZ(x, y, z, dir) d_sigmaZ[z + nz * (y + ny * (x + nx * dir))]

#define psi(x, y, z) d_psi[z + nz * ((y) + ny * (x))]
#define npsi(x, y, z) d_npsi[z + nz * ((y) + ny * (x))]

  const int tjj = threadIdx.y;
  const int tkk = threadIdx.x;

  // shift for each tile by updating device pointers
  d_psi = &(psi(XTILE * blockIdx.x, (BSIZE - 2) * blockIdx.y,
               (BSIZE - 2) * blockIdx.z));
  d_npsi = &(npsi(XTILE * blockIdx.x, (BSIZE - 2) * blockIdx.y,
                 (BSIZE - 2) * blockIdx.z));

  d_sigmaX = &(sigmaX(XTILE * blockIdx.x, (BSIZE - 2) * blockIdx.y,
                     (BSIZE - 2) * blockIdx.z, 0));
  d_sigmaY = &(sigmaY(XTILE * blockIdx.x, (BSIZE - 2) * blockIdx.y,
                     (BSIZE - 2) * blockIdx.z, 0));
  d_sigmaZ = &(sigmaZ(XTILE * blockIdx.x, (BSIZE - 2) * blockIdx.y,
                     (BSIZE - 2) * blockIdx.z, 0));

  int nLast_x = XTILE + 1;
  int nLast_y = (BSIZE - 1);
  int nLast_z = (BSIZE - 1);
  if (blockIdx.x == gridDim.x - 1)
    nLast_x = nx - 2 - XTILE * blockIdx.x + 1;
  if (blockIdx.y == gridDim.y - 1)
    nLast_y = ny - 2 - (BSIZE - 2) * blockIdx.y + 1;
  if (blockIdx.z == gridDim.z - 1)
    nLast_z = nz - 2 - (BSIZE - 2) * blockIdx.z + 1;

  if (tjj > nLast_y || tkk > nLast_z)
    return;

  // previous, current, next, and temp indices
  int pii, cii, nii, tii;
  pii = 0;
  cii = 1;
  nii = 2;

  sm_psi[cii][tjj][tkk] = psi(0, tjj, tkk);
  sm_psi[nii][tjj][tkk] = psi(1, tjj, tkk);
  Real xcharge, ycharge, zcharge, dV = 0;

  __syncthreads();

  // Initialize the incoming x-face flux before advancing through the tile.
  if ((tkk > 0) && (tkk < nLast_z) && (tjj > 0) && (tjj < nLast_y)) {
    Real xd = -V1(tjj, tkk) + V2(tjj, tkk);
    Real yd = (-V1(-1 + tjj, tkk) + V1(1 + tjj, tkk) - V2(-1 + tjj, tkk) +
               V2(1 + tjj, tkk)) /
              4.;
    Real zd = (-V1(tjj, -1 + tkk) + V1(tjj, 1 + tkk) - V2(tjj, -1 + tkk) +
               V2(tjj, 1 + tkk)) /
              4.;
    dV -= sigmaX(1, tjj, tkk, 0) * xd + sigmaX(1, tjj, tkk, 1) * yd +
          sigmaX(1, tjj, tkk, 2) * zd;
  }

  // Rotate plane indices to reuse storage as the window advances in x.
  tii = pii;
  pii = cii;
  cii = nii;
  nii = tii;

  for (int ii = 1; ii < nLast_x; ii++) {
    sm_psi[nii][tjj][tkk] = psi(ii + 1, tjj, tkk);
    __syncthreads();

    // y face current
    if ((tkk > 0) && (tkk < nLast_z) && (tjj < nLast_y)) {
      Real xd =
          (-V0(tjj, tkk) - V0(1 + tjj, tkk) + V2(tjj, tkk) + V2(1 + tjj, tkk)) /
          4.;
      Real yd = -V1(tjj, tkk) + V1(1 + tjj, tkk);
      Real zd = (-V1(tjj, -1 + tkk) + V1(tjj, 1 + tkk) - V1(1 + tjj, -1 + tkk) +
                 V1(1 + tjj, 1 + tkk)) /
                4.;
      ycharge = sigmaY(ii, tjj + 1, tkk, 0) * xd +
                sigmaY(ii, tjj + 1, tkk, 1) * yd +
                sigmaY(ii, tjj + 1, tkk, 2) * zd;
      dV += ycharge;
      sm_psi[3][tjj][tkk] = ycharge;
    }
    __syncthreads();

    if ((tkk > 0) && (tkk < nLast_z) && (tjj > 0) && (tjj < nLast_y))
      dV -= sm_psi[3][tjj - 1][tkk]; // bring from left

    __syncthreads();

    // z face current
    if ((tkk < nLast_z) && (tjj > 0) && (tjj < nLast_y)) {
      Real xd =
          (-V0(tjj, tkk) - V0(tjj, 1 + tkk) + V2(tjj, tkk) + V2(tjj, 1 + tkk)) /
          4.;
      Real yd = (-V1(-1 + tjj, tkk) - V1(-1 + tjj, 1 + tkk) + V1(1 + tjj, tkk) +
                 V1(1 + tjj, 1 + tkk)) /
                4.;
      Real zd = -V1(tjj, tkk) + V1(tjj, 1 + tkk);
      zcharge = sigmaZ(ii, tjj, tkk + 1, 0) * xd +
                sigmaZ(ii, tjj, tkk + 1, 1) * yd +
                sigmaZ(ii, tjj, tkk + 1, 2) * zd;
      dV += zcharge;
      sm_psi[3][tjj][tkk] = zcharge;
    }

    __syncthreads();

    if ((tkk > 0) && (tkk < nLast_z) && (tjj > 0) && (tjj < nLast_y))
      dV -= sm_psi[3][tjj][tkk - 1];
    __syncthreads();

    // x face current
    if ((tkk > 0) && (tkk < nLast_z) && (tjj > 0) && (tjj < nLast_y)) {
      Real xd = -V1(tjj, tkk) + V2(tjj, tkk);
      Real yd = (-V1(-1 + tjj, tkk) + V1(1 + tjj, tkk) - V2(-1 + tjj, tkk) +
                 V2(1 + tjj, tkk)) /
                4.;
      Real zd = (-V1(tjj, -1 + tkk) + V1(tjj, 1 + tkk) - V2(tjj, -1 + tkk) +
                 V2(tjj, 1 + tkk)) /
                4.;
      xcharge = sigmaX(ii + 1, tjj, tkk, 0) * xd +
                sigmaX(ii + 1, tjj, tkk, 1) * yd +
                sigmaX(ii + 1, tjj, tkk, 2) * zd;
      dV += xcharge;
      npsi(ii, tjj, tkk) = dV; // store dV
      dV = -xcharge;           // pass to the next cell in x-dir
    }
    __syncthreads();
    tii = pii;
    pii = cii;
    cii = nii;
    nii = tii;
  }
}

// These indexing macros belong only to the adapted kernel.
#undef V0
#undef V1
#undef V2
#undef sigmaX
#undef sigmaY
#undef sigmaZ
#undef psi
#undef npsi
