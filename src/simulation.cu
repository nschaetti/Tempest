#include "tempest/simulation.hpp"

__global__ void wave_step_kernel(
    float* p_new,
    const float* p,
    const float* p_old,
    const float* c2,
    int nx,
    int nz,
    float dt2dx2
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float tile[BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];

    int lx = threadIdx.x + 1;
    int lz = threadIdx.y + 1;

    if (ix < nx && iz < nz)
        tile[lz][lx] = p[iz * nx + ix];

    if (threadIdx.x == 0 && ix > 0)
        tile[lz][lx - 1] = p[iz * nx + (ix - 1)];
    if (threadIdx.x == blockDim.x - 1 && ix < nx - 1)
        tile[lz][lx + 1] = p[iz * nx + (ix + 1)];
    if (threadIdx.y == 0 && iz > 0)
        tile[lz - 1][lx] = p[(iz - 1) * nx + ix];
    if (threadIdx.y == blockDim.y - 1 && iz < nz - 1)
        tile[lz + 1][lx] = p[(iz + 1) * nx + ix];

    __syncthreads();

    if (ix > 0 && ix < nx - 1 && iz > 0 && iz < nz - 1) {
        float laplacian =
            tile[lz][lx - 1] + tile[lz][lx + 1] +
            tile[lz - 1][lx] + tile[lz + 1][lx] -
            4.0f * tile[lz][lx];

        int idx = iz * nx + ix;
        p_new[idx] = 2.0f * tile[lz][lx] - p_old[idx]
                   + c2[idx] * dt2dx2 * laplacian;
    }
}

__global__ void init_wavefield_kernel(float* p_old, float* p, float* p_new, float* c2,
                                      float c0sq, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        p_old[idx] = 0.0f;
        p[idx] = 0.0f;
        p_new[idx] = 0.0f;
        c2[idx] = c0sq;
    }
}

__global__ void zero_wavefield_kernel(float* p_old, float* p, float* p_new, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        p_old[idx] = 0.0f;
        p[idx] = 0.0f;
        p_new[idx] = 0.0f;
    }
}

__global__ void point_source_kernel(float* p, int index, float amplitude) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        p[index] = amplitude;
    }
}

void initialize_wavefields(float* d_p_old,
                           float* d_p,
                           float* d_p_new,
                           float* d_c2,
                           float c0sq,
                           int n,
                           int source_index,
                           float amplitude) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    init_wavefield_kernel<<<blocks, threads>>>(d_p_old, d_p, d_p_new, d_c2, c0sq, n);
    CUDA_CHECK(cudaGetLastError());
    point_source_kernel<<<1, 1>>>(d_p, source_index, amplitude);
    CUDA_CHECK(cudaGetLastError());
}

void reset_wavefields(float* d_p_old,
                      float* d_p,
                      float* d_p_new,
                      int n,
                      int source_index,
                      float amplitude) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    zero_wavefield_kernel<<<blocks, threads>>>(d_p_old, d_p, d_p_new, n);
    CUDA_CHECK(cudaGetLastError());
    point_source_kernel<<<1, 1>>>(d_p, source_index, amplitude);
    CUDA_CHECK(cudaGetLastError());
}
