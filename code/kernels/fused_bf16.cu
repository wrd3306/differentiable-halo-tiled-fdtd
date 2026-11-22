// Mixed-precision tiled 3-D FDTD kernels (L4): bf16 field storage, fp32 math.
//
// The six field components live in bfloat16 (halving their global and shared
// memory traffic). Every value is promoted to fp32 before use; all differences,
// curl, psi and source arithmetic run in fp32, and results are rounded back to
// bf16 only on store. CPML psi and coefficient arrays remain fp32 for stability.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define BX 8
#define BY 8
#define BZ 8
#define SX (BX + 1)
#define SY (BY + 1)
#define SZ (BZ + 1)
#define NSH (SX * SY * SZ)
#define SIDX(x, y, z) (((x) * SY + (y)) * SZ + (z))

using b16 = __nv_bfloat16;

__device__ inline long gidx(int i, int j, int k, int Ny, int Nz) {
  return ((long)i * Ny + j) * Nz + k;
}

__device__ inline int clampi(int v, int n) {
  return v < 0 ? 0 : (v >= n ? n - 1 : v);
}

__device__ void load_tile(b16* sh, const b16* g, int x0, int y0, int z0,
                          int Nx, int Ny, int Nz) {
  int tid = threadIdx.x + threadIdx.y * BX + threadIdx.z * BX * BY;
  int nth = BX * BY * BZ;
  for (int p = tid; p < NSH; p += nth) {
    int z = p % SZ, y = (p / SZ) % SY, x = p / (SY * SZ);
    sh[p] = g[gidx(clampi(x0 + x, Nx), clampi(y0 + y, Ny),
                   clampi(z0 + z, Nz), Ny, Nz)];
  }
}

__global__ void h_bf16_kernel(
    const b16* __restrict__ Ex, const b16* __restrict__ Ey,
    const b16* __restrict__ Ez, const float* __restrict__ bz,
    const float* __restrict__ cz, b16* __restrict__ Hx, b16* __restrict__ Hy,
    b16* __restrict__ Hz, float* __restrict__ pe, float* __restrict__ px,
    int Nx, int Ny, int Nz, float idx, float idy, float idz, float mh) {
  __shared__ b16 sEx[NSH], sEy[NSH], sEz[NSH];
  int x0 = blockIdx.x * BX, y0 = blockIdx.y * BY, z0 = blockIdx.z * BZ;
  load_tile(sEx, Ex, x0, y0, z0, Nx, Ny, Nz);
  load_tile(sEy, Ey, x0, y0, z0, Nx, Ny, Nz);
  load_tile(sEz, Ez, x0, y0, z0, Nx, Ny, Nz);
  __syncthreads();

  int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;
  int gi = x0 + x, gj = y0 + y, gk = z0 + z;
  if (gi >= Nx || gj >= Ny || gk >= Nz) return;
  long s = gidx(gi, gj, gk, Ny, Nz);

  float ex = __bfloat162float(sEx[SIDX(x, y, z)]);
  float ey = __bfloat162float(sEy[SIDX(x, y, z)]);
  float ez = __bfloat162float(sEz[SIDX(x, y, z)]);
  float dEydz = (__bfloat162float(sEy[SIDX(x, y, z + 1)]) - ey) * idz;
  float dExdz = (__bfloat162float(sEx[SIDX(x, y, z + 1)]) - ex) * idz;
  float dEzdy = (__bfloat162float(sEz[SIDX(x, y + 1, z)]) - ez) * idy;
  float dEzdx = (__bfloat162float(sEz[SIDX(x + 1, y, z)]) - ez) * idx;
  float dEydx = (__bfloat162float(sEy[SIDX(x + 1, y, z)]) - ey) * idx;
  float dExdy = (__bfloat162float(sEx[SIDX(x, y + 1, z)]) - ex) * idy;

  float b = bz[s], c = cz[s];
  float pev = b * pe[s] + c * dEydz;
  float pxv = b * px[s] + c * dExdz;
  pe[s] = pev;
  px[s] = pxv;

  float hx = __bfloat162float(Hx[s]) - mh * (dEzdy - dEydz - pev);
  float hy = __bfloat162float(Hy[s]) - mh * (dExdz + pxv - dEzdx);
  float hz = __bfloat162float(Hz[s]) - mh * (dEydx - dExdy);
  Hx[s] = __float2bfloat16(hx);
  Hy[s] = __float2bfloat16(hy);
  Hz[s] = __float2bfloat16(hz);
}

__global__ void e_bf16_kernel(
    b16* __restrict__ Ex, b16* __restrict__ Ey, b16* __restrict__ Ez,
    const b16* __restrict__ Hx, const b16* __restrict__ Hy,
    const b16* __restrict__ Hz, const float* __restrict__ Ce,
    const float* __restrict__ De, const float* __restrict__ bz,
    const float* __restrict__ cz, float* __restrict__ phy,
    float* __restrict__ phx, const float* __restrict__ jprof,
    const float* __restrict__ mx, const float* __restrict__ my, int Nx,
    int Ny, int Nz, float idx, float idy, float idz, float sv) {
  __shared__ b16 sHx[NSH], sHy[NSH], sHz[NSH];
  int x0 = blockIdx.x * BX - 1, y0 = blockIdx.y * BY - 1, z0 = blockIdx.z * BZ - 1;
  load_tile(sHx, Hx, x0, y0, z0, Nx, Ny, Nz);
  load_tile(sHy, Hy, x0, y0, z0, Nx, Ny, Nz);
  load_tile(sHz, Hz, x0, y0, z0, Nx, Ny, Nz);
  __syncthreads();

  int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;
  int gi = x0 + 1 + x, gj = y0 + 1 + y, gk = z0 + 1 + z;
  if (gi >= Nx || gj >= Ny || gk >= Nz) return;
  long s = gidx(gi, gj, gk, Ny, Nz);

  float hx = __bfloat162float(sHx[SIDX(x + 1, y + 1, z + 1)]);
  float hy = __bfloat162float(sHy[SIDX(x + 1, y + 1, z + 1)]);
  float hz = __bfloat162float(sHz[SIDX(x + 1, y + 1, z + 1)]);
  float dHzdy = (hz - __bfloat162float(sHz[SIDX(x + 1, y, z + 1)])) * idy;
  float dHxdy = (hx - __bfloat162float(sHx[SIDX(x + 1, y, z + 1)])) * idy;
  float dHydz = (hy - __bfloat162float(sHy[SIDX(x + 1, y + 1, z)])) * idz;
  float dHxdz = (hx - __bfloat162float(sHx[SIDX(x + 1, y + 1, z)])) * idz;
  float dHzdx = (hz - __bfloat162float(sHz[SIDX(x, y + 1, z + 1)])) * idx;
  float dHydx = (hy - __bfloat162float(sHy[SIDX(x, y + 1, z + 1)])) * idx;

  float b = bz[s], c = cz[s];
  float phyv = b * phy[s] + c * dHydz;
  float phxv = b * phx[s] + c * dHxdz;
  phy[s] = phyv;
  phx[s] = phxv;

  float de = De[s];
  float ex = Ce[s] * __bfloat162float(Ex[s]) + de * (dHzdy - dHydz - phyv);
  float ey = Ce[s] * __bfloat162float(Ey[s]) + de * (dHxdz + phxv - dHzdx);
  float ez = Ce[s] * __bfloat162float(Ez[s]) + de * (dHydx - dHxdy);
  ey -= de * sv * jprof[s];
  Ex[s] = __float2bfloat16(ex * my[s]);
  Ey[s] = __float2bfloat16(ey * mx[s]);
  Ez[s] = __float2bfloat16(ez * mx[s] * my[s]);
}

dim3 tgrid(int Nx, int Ny, int Nz) {
  return dim3((Nx + BX - 1) / BX, (Ny + BY - 1) / BY, (Nz + BZ - 1) / BZ);
}

void h_launch_bf16(torch::Tensor Ex, torch::Tensor Ey, torch::Tensor Ez,
                   torch::Tensor bz, torch::Tensor cz, torch::Tensor Hx,
                   torch::Tensor Hy, torch::Tensor Hz, torch::Tensor pe,
                   torch::Tensor px, int Nx, int Ny, int Nz, float idx,
                   float idy, float idz, float mh) {
  h_bf16_kernel<<<tgrid(Nx, Ny, Nz), dim3(BX, BY, BZ)>>>(
      (const b16*)Ex.data_ptr(), (const b16*)Ey.data_ptr(),
      (const b16*)Ez.data_ptr(), bz.data_ptr<float>(), cz.data_ptr<float>(),
      (b16*)Hx.data_ptr(), (b16*)Hy.data_ptr(), (b16*)Hz.data_ptr(),
      pe.data_ptr<float>(), px.data_ptr<float>(), Nx, Ny, Nz, idx, idy, idz,
      mh);
}

void e_launch_bf16(torch::Tensor Ex, torch::Tensor Ey, torch::Tensor Ez,
                   torch::Tensor Hx, torch::Tensor Hy, torch::Tensor Hz,
                   torch::Tensor Ce, torch::Tensor De, torch::Tensor bz,
                   torch::Tensor cz, torch::Tensor phy, torch::Tensor phx,
                   torch::Tensor jprof, torch::Tensor mx, torch::Tensor my,
                   int Nx, int Ny, int Nz, float idx, float idy, float idz,
                   float sv) {
  e_bf16_kernel<<<tgrid(Nx, Ny, Nz), dim3(BX, BY, BZ)>>>(
      (b16*)Ex.data_ptr(), (b16*)Ey.data_ptr(), (b16*)Ez.data_ptr(),
      (const b16*)Hx.data_ptr(), (const b16*)Hy.data_ptr(),
      (const b16*)Hz.data_ptr(), Ce.data_ptr<float>(), De.data_ptr<float>(),
      bz.data_ptr<float>(), cz.data_ptr<float>(), phy.data_ptr<float>(),
      phx.data_ptr<float>(), jprof.data_ptr<float>(), mx.data_ptr<float>(),
      my.data_ptr<float>(), Nx, Ny, Nz, idx, idy, idz, sv);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("h_launch_bf16", &h_launch_bf16);
  m.def("e_launch_bf16", &e_launch_bf16);
}
