#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// ======================================================================
// GPU Kernel: compute one time step of 2D acoustic wave propagation
// ======================================================================
__global__ void wave_step_kernel(
    float* p_new, const float* p, const float* p_old,
    const float* c2, int nx, int nz,
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

// ======================================================================
// Host code
// ======================================================================
int main() {
    const int nx = 10000;
    const int nz = 10000;
    const int n = nx * nz;
    const float c0 = 1500.0f;
    const float dt = 0.001f;
    const float dx = 1.0f;
    const float dt2dx2 = (dt * dt) / (dx * dx);
    const int nt = 1000;

    float *h_p_old = new float[n];
    float *h_p = new float[n];
    float *h_p_new = new float[n];
    float *h_c2 = new float[n];

    for (int i = 0; i < n; ++i) {
        h_p_old[i] = 0.0f;
        h_p[i] = 0.0f;
        h_c2[i] = c0 * c0;
    }

    int sx = nx / 2;
    int sz = nz / 2;
    h_p[sz * nx + sx] = 1.0f;

    float *d_p_old, *d_p, *d_p_new, *d_c2;
    cudaMalloc(&d_p_old, n * sizeof(float));
    cudaMalloc(&d_p, n * sizeof(float));
    cudaMalloc(&d_p_new, n * sizeof(float));
    cudaMalloc(&d_c2, n * sizeof(float));

    cudaMemcpy(d_p_old, h_p_old, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, h_p, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c2, h_c2, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
              (nz + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    // ============================================================
    // ⏱️ Mesure du temps avec CUDA Events
    // ============================================================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);  // début de la mesure GPU

    for (int it = 0; it < nt; ++it) {
        wave_step_kernel<<<grid, block>>>(d_p_new, d_p, d_p_old, d_c2, nx, nz, dt2dx2);
        cudaDeviceSynchronize();
        std::swap(d_p_old, d_p);
        std::swap(d_p, d_p_new);
    }

    cudaEventRecord(stop);   // fin de la mesure
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "⏱️ Simulation time (" << nt << " steps, "
              << nx << "x" << nz << " grid): "
              << milliseconds << " ms  ≈  "
              << milliseconds / 1000.0f << " s" << std::endl;

    cudaMemcpy(h_p_new, d_p_new, n * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Centre: " << h_p_new[sz * nx + sx] << std::endl;

    // Nettoyage
    delete[] h_p_old;
    delete[] h_p;
    delete[] h_p_new;
    delete[] h_c2;
    cudaFree(d_p_old);
    cudaFree(d_p);
    cudaFree(d_p_new);
    cudaFree(d_c2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
