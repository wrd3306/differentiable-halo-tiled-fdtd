// Fused 3-D vector FDTD update kernels (L3a: register-level fusion).
//
// One thread per Yee cell. Spatial derivatives are computed directly from
// neighbour reads, so the 12 derivative tensors of the naive engine are never
// materialised in global memory. CPML psi, the TE10 source and PEC walls are
// folded into the same per-cell arithmetic used by the L0 reference, so the
// fused engine is a drop-in replacement and can be checked field-by-field.
//
// All tensors are contiguous, shape (Nx,Ny,Nz), fp32.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define IDX3(i, j, k) (((i) * Ny + (j)) * Nz + (k))

// Fused magnetic half-step: updates Hx,Hy,Hz and psi pe,px.
__global__ void h_update_kernel(
    const float* __restrict__ Ex, const float* __restrict__ Ey,
    const float* __restrict__ Ez, const float* __restrict__ bz,
    const float* __restrict__ cz, float* __restrict__ Hx,
    float* __restrict__ Hy, float* __restrict__ Hz,
    float* __restrict__ pe, float* __restrict__ px, int Nx, int Ny, int Nz,
    float idx, float idy, float idz, float mh) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= Nx || j >= Ny || k >= Nz) return;
  long s = IDX3(i, j, k);

  // E forward differences with L0 edge rules (zero on the last row).
  float dEydz = 0.f, dExdz = 0.f;
  if (k < Nz - 1) {
    dEydz = (Ey[s + 1] - Ey[s]) * idz;
    dExdz = (Ex[s + 1] - Ex[s]) * idz;
  }
  float dEzdy = 0.f;
  if (j < Ny - 1) dEzdy = (Ez[IDX3(i, j + 1, k)] - Ez[s]) * idy;
  float dEzdx = 0.f;
  if (i < Nx - 1) dEzdx = (Ez[IDX3(i + 1, j, k)] - Ez[s]) * idx;
  float dEydx = 0.f;
  if (i < Nx - 1) dEydx = (Ey[IDX3(i + 1, j, k)] - Ey[s]) * idx;
  float dExdy = 0.f;
  if (j < Ny - 1) dExdy = (Ex[IDX3(i, j + 1, k)] - Ex[s]) * idy;

  // CPML psi.
  float b = bz[s], c = cz[s];
  float pev = b * pe[s] + c * dEydz;
  float pxv = b * px[s] + c * dExdz;
  pe[s] = pev;
  px[s] = pxv;

  // H update.
  Hx[s] = Hx[s] - mh * (dEzdy - dEydz - pev);
  Hy[s] = Hy[s] - mh * (dExdz + pxv - dEzdx);
  Hz[s] = Hz[s] - mh * (dEydx - dExdy);
}

// Fused electric half-step: updates Ex,Ey,Ez and psi phy,phx, then source/PEC.
__global__ void e_update_kernel(
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz, const float* __restrict__ Ce,
    const float* __restrict__ De, const float* __restrict__ bz,
    const float* __restrict__ cz, float* __restrict__ phy,
    float* __restrict__ phx, const float* __restrict__ jprof,
    const float* __restrict__ mx, const float* __restrict__ my, int Nx,
    int Ny, int Nz, float idx, float idy, float idz, float sv) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= Nx || j >= Ny || k >= Nz) return;
  long s = IDX3(i, j, k);

  // H backward differences with L0 edge rules (zero on the first row).
  float dHzdy = 0.f, dHxdy = 0.f;
  if (j > 0) {
    dHzdy = (Hz[s] - Hz[IDX3(i, j - 1, k)]) * idy;
    dHxdy = (Hx[s] - Hx[IDX3(i, j - 1, k)]) * idy;
  }
  float dHydz = 0.f, dHxdz = 0.f;
  if (k > 0) {
    dHydz = (Hy[s] - Hy[s - 1]) * idz;
    dHxdz = (Hx[s] - Hx[s - 1]) * idz;
  }
  float dHzdx = 0.f;
  if (i > 0) dHzdx = (Hz[s] - Hz[IDX3(i - 1, j, k)]) * idx;
  float dHydx = 0.f;
  if (i > 0) dHydx = (Hy[s] - Hy[IDX3(i - 1, j, k)]) * idx;

  // CPML psi.
  float b = bz[s], c = cz[s];
  float phyv = b * phy[s] + c * dHydz;
  float phxv = b * phx[s] + c * dHxdz;
  phy[s] = phyv;
  phx[s] = phxv;

  // E update.
  float de = De[s];
  float ex = Ce[s] * Ex[s] + de * (dHzdy - dHydz - phyv);
  float ey = Ce[s] * Ey[s] + de * (dHxdz + phxv - dHzdx);
  float ez = Ce[s] * Ez[s] + de * (dHydx - dHxdy);

  // Source (jprof is zero except on the source plane), then PEC masks.
  ey -= de * sv * jprof[s];
  Ex[s] = ex * my[s];
  Ey[s] = ey * mx[s];
  Ez[s] = ez * mx[s] * my[s];
}

dim3 grid3(int Nx, int Ny, int Nz, dim3 block) {
  return dim3((Nx + block.x - 1) / block.x, (Ny + block.y - 1) / block.y,
              (Nz + block.z - 1) / block.z);
}

void h_launch(torch::Tensor Ex, torch::Tensor Ey, torch::Tensor Ez,
              torch::Tensor bz, torch::Tensor cz, torch::Tensor Hx,
              torch::Tensor Hy, torch::Tensor Hz, torch::Tensor pe,
              torch::Tensor px, int Nx, int Ny, int Nz, float idx, float idy,
              float idz, float mh) {
  dim3 block(8, 8, 4);
  dim3 grid = grid3(Nx, Ny, Nz, block);
  h_update_kernel<<<grid, block>>>(
      Ex.data_ptr<float>(), Ey.data_ptr<float>(), Ez.data_ptr<float>(),
      bz.data_ptr<float>(), cz.data_ptr<float>(), Hx.data_ptr<float>(),
      Hy.data_ptr<float>(), Hz.data_ptr<float>(), pe.data_ptr<float>(),
      px.data_ptr<float>(), Nx, Ny, Nz, idx, idy, idz, mh);
}

void e_launch(torch::Tensor Ex, torch::Tensor Ey, torch::Tensor Ez,
              torch::Tensor Hx, torch::Tensor Hy, torch::Tensor Hz,
              torch::Tensor Ce, torch::Tensor De, torch::Tensor bz,
              torch::Tensor cz, torch::Tensor phy, torch::Tensor phx,
              torch::Tensor jprof, torch::Tensor mx, torch::Tensor my, int Nx,
              int Ny, int Nz, float idx, float idy, float idz, float sv) {
  dim3 block(8, 8, 4);
  dim3 grid = grid3(Nx, Ny, Nz, block);
  e_update_kernel<<<grid, block>>>(
      Ex.data_ptr<float>(), Ey.data_ptr<float>(), Ez.data_ptr<float>(),
      Hx.data_ptr<float>(), Hy.data_ptr<float>(), Hz.data_ptr<float>(),
      Ce.data_ptr<float>(), De.data_ptr<float>(), bz.data_ptr<float>(),
      cz.data_ptr<float>(), phy.data_ptr<float>(), phx.data_ptr<float>(),
      jprof.data_ptr<float>(), mx.data_ptr<float>(), my.data_ptr<float>(), Nx,
      Ny, Nz, idx, idy, idz, sv);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("h_launch", &h_launch);
  m.def("e_launch", &e_launch);
}
