#include "tempest/simulation.hpp"

//------------------------------------------------------------------------------
// CUDA kernels backing the finite-difference solver
//------------------------------------------------------------------------------

/**
 * Each block processes a BLOCK_SIZE_X by BLOCK_SIZE_Y tile of the pressure field.
 * We keep a 1-cell halo in shared memory so we can compute the Laplacian using
 * only fast on-chip memory instead of re-loading neighbours from global memory.
 */
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

    // Load the halo cells needed for the 5-point stencil. Each conditional is
    // guarded to avoid reading outside the global domain.
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
        // Standard second-order wave equation discretization:
        // p_new = 2*p - p_old + c^2 * dt^2/dx^2 * laplacian.
        p_new[idx] = 2.0f * tile[lz][lx] - p_old[idx]
                   + c2[idx] * dt2dx2 * laplacian;
    }
}

// Initialize pressure fields to zero and c^2 to a constant reference value.
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

// Same as init_wavefield_kernel but without touching c^2 (which is constant).
__global__ void zero_wavefield_kernel(float* p_old, float* p, float* p_new, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        p_old[idx] = 0.0f;
        p[idx] = 0.0f;
        p_new[idx] = 0.0f;
    }
}

// Inject a single point source by writing the requested amplitude once.
__global__ void point_source_kernel(float* p, int index, float amplitude) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        p[index] = amplitude;
    }
}

//------------------------------------------------------------------------------
// Host helpers that launch the kernels above
//------------------------------------------------------------------------------

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
