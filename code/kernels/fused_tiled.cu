// Shared-memory tiled 3-D vector FDTD kernels (L3b).
//
// A thread block owns an inner tile (BX x BY x BZ). The field components over
// the (BX+1)x(BY+1)x(BZ+1) halo are cooperatively loaded into shared memory,
// so each global value is read once and reused by neighbouring threads instead
// of being re-read by up to six cells. Out-of-range halo coordinates clamp to
// the domain edge, which makes the boundary difference zero, reproducing the
// L0 forward/backward edge rules. Per-cell psi / coefficients / output fields
// are touched once directly from global memory.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BX 8
#define BY 8
#define BZ 8
#define SX (BX + 1)
#define SY (BY + 1)
#define SZ (BZ + 1)
#define NSH (SX * SY * SZ)
#define SIDX(x, y, z) (((x) * SY + (y)) * SZ + (z))

__device__ inline long gidx(int i, int j, int k, int Ny, int Nz) {
  return ((long)i * Ny + j) * Nz + k;
}

__device__ inline int clampi(int v, int n) {
  return v < 0 ? 0 : (v >= n ? n - 1 : v);
}

// Load one component's halo tile into shared memory; clamp out-of-range reads.
__device__ void load_tile(float* sh, const float* g, int x0, int y0, int z0,
                          int Nx, int Ny, int Nz) {
  int tid = threadIdx.x + threadIdx.y * BX + threadIdx.z * BX * BY;
  int nth = BX * BY * BZ;
  for (int p = tid; p < NSH; p += nth) {
    int z = p % SZ;
    int y = (p / SZ) % SY;
    int x = p / (SY * SZ);
    int gi = clampi(x0 + x, Nx), gj = clampi(y0 + y, Ny),
        gk = clampi(z0 + z, Nz);
    sh[p] = g[gidx(gi, gj, gk, Ny, Nz)];
  }
}

__global__ void h_tiled_kernel(
    const float* __restrict__ Ex, const float* __restrict__ Ey,
    const float* __restrict__ Ez, const float* __restrict__ bz,
    const float* __restrict__ cz, float* __restrict__ Hx,
    float* __restrict__ Hy, float* __restrict__ Hz,
    float* __restrict__ pe, float* __restrict__ px, int Nx, int Ny, int Nz,
    float idx, float idy, float idz, float mh) {
  __shared__ float sEx[NSH], sEy[NSH], sEz[NSH];
  int x0 = blockIdx.x * BX, y0 = blockIdx.y * BY, z0 = blockIdx.z * BZ;
  load_tile(sEx, Ex, x0, y0, z0, Nx, Ny, Nz);
  load_tile(sEy, Ey, x0, y0, z0, Nx, Ny, Nz);
  load_tile(sEz, Ez, x0, y0, z0, Nx, Ny, Nz);
  __syncthreads();

  int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;
  int gi = x0 + x, gj = y0 + y, gk = z0 + z;
  if (gi >= Nx || gj >= Ny || gk >= Nz) return;
  long s = gidx(gi, gj, gk, Ny, Nz);

  // Forward differences from shared halo (clamped edges -> zero at boundary).
  float dEydz = (sEy[SIDX(x, y, z + 1)] - sEy[SIDX(x, y, z)]) * idz;
  float dExdz = (sEx[SIDX(x, y, z + 1)] - sEx[SIDX(x, y, z)]) * idz;
  float dEzdy = (sEz[SIDX(x, y + 1, z)] - sEz[SIDX(x, y, z)]) * idy;
  float dEzdx = (sEz[SIDX(x + 1, y, z)] - sEz[SIDX(x, y, z)]) * idx;
  float dEydx = (sEy[SIDX(x + 1, y, z)] - sEy[SIDX(x, y, z)]) * idx;
  float dExdy = (sEx[SIDX(x, y + 1, z)] - sEx[SIDX(x, y, z)]) * idy;

  float b = bz[s], c = cz[s];
  float pev = b * pe[s] + c * dEydz;
  float pxv = b * px[s] + c * dExdz;
  pe[s] = pev;
  px[s] = pxv;

  Hx[s] = Hx[s] - mh * (dEzdy - dEydz - pev);
  Hy[s] = Hy[s] - mh * (dExdz + pxv - dEzdx);
  Hz[s] = Hz[s] - mh * (dEydx - dExdy);
}

__global__ void e_tiled_kernel(
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz, const float* __restrict__ Ce,
    const float* __restrict__ De, const float* __restrict__ bz,
    const float* __restrict__ cz, float* __restrict__ phy,
    float* __restrict__ phx, const float* __restrict__ jprof,
    const float* __restrict__ mx, const float* __restrict__ my, int Nx,
    int Ny, int Nz, float idx, float idy, float idz, float sv) {
  __shared__ float sHx[NSH], sHy[NSH], sHz[NSH];
  // Backward differences: halo lies one cell below, so shift the base.
  int x0 = blockIdx.x * BX - 1, y0 = blockIdx.y * BY - 1, z0 = blockIdx.z * BZ - 1;
  load_tile(sHx, Hx, x0, y0, z0, Nx, Ny, Nz);
  load_tile(sHy, Hy, x0, y0, z0, Nx, Ny, Nz);
  load_tile(sHz, Hz, x0, y0, z0, Nx, Ny, Nz);
  __syncthreads();

  int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;
  int gi = x0 + 1 + x, gj = y0 + 1 + y, gk = z0 + 1 + z;
  if (gi >= Nx || gj >= Ny || gk >= Nz) return;
  long s = gidx(gi, gj, gk, Ny, Nz);

  // Backward differences (shared index 0 is the lower neighbour).
  float dHzdy = (sHz[SIDX(x + 1, y + 1, z + 1)] - sHz[SIDX(x + 1, y, z + 1)]) * idy;
  float dHxdy = (sHx[SIDX(x + 1, y + 1, z + 1)] - sHx[SIDX(x + 1, y, z + 1)]) * idy;
  float dHydz = (sHy[SIDX(x + 1, y + 1, z + 1)] - sHy[SIDX(x + 1, y + 1, z)]) * idz;
  float dHxdz = (sHx[SIDX(x + 1, y + 1, z + 1)] - sHx[SIDX(x + 1, y + 1, z)]) * idz;
  float dHzdx = (sHz[SIDX(x + 1, y + 1, z + 1)] - sHz[SIDX(x, y + 1, z + 1)]) * idx;
  float dHydx = (sHy[SIDX(x + 1, y + 1, z + 1)] - sHy[SIDX(x, y + 1, z + 1)]) * idx;

  float b = bz[s], c = cz[s];
  float phyv = b * phy[s] + c * dHydz;
  float phxv = b * phx[s] + c * dHxdz;
  phy[s] = phyv;
  phx[s] = phxv;

  float de = De[s];
  float ex = Ce[s] * Ex[s] + de * (dHzdy - dHydz - phyv);
  float ey = Ce[s] * Ey[s] + de * (dHxdz + phxv - dHzdx);
  float ez = Ce[s] * Ez[s] + de * (dHydx - dHxdy);
  ey -= de * sv * jprof[s];
  Ex[s] = ex * my[s];
  Ey[s] = ey * mx[s];
  Ez[s] = ez * mx[s] * my[s];
}

dim3 tgrid(int Nx, int Ny, int Nz) {
  return dim3((Nx + BX - 1) / BX, (Ny + BY - 1) / BY, (Nz + BZ - 1) / BZ);
}

void h_launch_tiled(torch::Tensor Ex, torch::Tensor Ey, torch::Tensor Ez,
                    torch::Tensor bz, torch::Tensor cz, torch::Tensor Hx,
                    torch::Tensor Hy, torch::Tensor Hz, torch::Tensor pe,
                    torch::Tensor px, int Nx, int Ny, int Nz, float idx,
                    float idy, float idz, float mh) {
  h_tiled_kernel<<<tgrid(Nx, Ny, Nz), dim3(BX, BY, BZ)>>>(
      Ex.data_ptr<float>(), Ey.data_ptr<float>(), Ez.data_ptr<float>(),
      bz.data_ptr<float>(), cz.data_ptr<float>(), Hx.data_ptr<float>(),
      Hy.data_ptr<float>(), Hz.data_ptr<float>(), pe.data_ptr<float>(),
      px.data_ptr<float>(), Nx, Ny, Nz, idx, idy, idz, mh);
}

void e_launch_tiled(torch::Tensor Ex, torch::Tensor Ey, torch::Tensor Ez,
                    torch::Tensor Hx, torch::Tensor Hy, torch::Tensor Hz,
                    torch::Tensor Ce, torch::Tensor De, torch::Tensor bz,
                    torch::Tensor cz, torch::Tensor phy, torch::Tensor phx,
                    torch::Tensor jprof, torch::Tensor mx, torch::Tensor my,
                    int Nx, int Ny, int Nz, float idx, float idy, float idz,
                    float sv) {
  e_tiled_kernel<<<tgrid(Nx, Ny, Nz), dim3(BX, BY, BZ)>>>(
      Ex.data_ptr<float>(), Ey.data_ptr<float>(), Ez.data_ptr<float>(),
      Hx.data_ptr<float>(), Hy.data_ptr<float>(), Hz.data_ptr<float>(),
      Ce.data_ptr<float>(), De.data_ptr<float>(), bz.data_ptr<float>(),
      cz.data_ptr<float>(), phy.data_ptr<float>(), phx.data_ptr<float>(),
      jprof.data_ptr<float>(), mx.data_ptr<float>(), my.data_ptr<float>(), Nx,
      Ny, Nz, idx, idy, idz, sv);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("h_launch_tiled", &h_launch_tiled);
  m.def("e_launch_tiled", &e_launch_tiled);
}
