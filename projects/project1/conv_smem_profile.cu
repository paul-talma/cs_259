#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

typedef float VTYPE;

#define KY 3
#define KX 3
#define TILE_N   32
#define TILE_SP  8
#define NI_CHUNK 64

#define INPUT_IDX_3D(b, y, x, ni, NYPAD, NXPAD, Ni) \
    ((((y) * (NXPAD) + (x)) * (Ni)) + (ni) + (b) * (NYPAD) * (NXPAD) * (Ni))

#define OUTPUT_IDX_3D(b, y, x, nn, NYSCL, NXSCL, Nn) \
    ((((y) * (NXSCL) + (x)) * (Nn)) + (nn) + (b) * (NYSCL) * (NXSCL) * (Nn))

#define WEIGHT_IDX(ky, kx, nn, ni, KX, Nn, Ni) \
    ((((ky) * (KX) + (kx)) * (Nn) + (nn)) * (Ni) + (ni))

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " -> " << cudaGetErrorString(err__) << std::endl;      \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

__device__ float relu(float x) { return x > 0.f ? x : 0.f; }

__global__ void conv2d_smem(const float *weight,
                            const float *input,
                            float *output,
                            int B,
                            int Ny,
                            int Nx,
                            int Ni,
                            int Nn,
                            int NYPAD,
                            int NXPAD,
                            int NYSCL,
                            int NXSCL) {

    extern __shared__ float smem[];

    int col      = blockIdx.x;
    int row      = blockIdx.y * TILE_SP + threadIdx.y;
    int Nn_tiles = (Nn + TILE_N - 1) / TILE_N;
    int nn_tile  = blockIdx.z % Nn_tiles;
    int b        = blockIdx.z / Nn_tiles;
    int nn       = nn_tile * TILE_N + threadIdx.x;
    bool valid   = (nn < Nn) && (row < NYSCL);

    int tid           = threadIdx.y * TILE_N + threadIdx.x;
    int block_threads = TILE_N * TILE_SP;

    float sum = 0.f;

    for (int ky = 0; ky < KY; ky++) {
        for (int kx = 0; kx < KX; kx++) {
            for (int ni_base = 0; ni_base < Ni; ni_base += NI_CHUNK) {
                int chunk = (ni_base + NI_CHUNK <= Ni) ? NI_CHUNK : (Ni - ni_base);

                for (int s = tid; s < TILE_N * chunk; s += block_threads) {
                    int nn_off    = s / chunk;
                    int ni_off    = s % chunk;
                    int global_nn = nn_tile * TILE_N + nn_off;
                    smem[nn_off * (NI_CHUNK + 1) + ni_off] =
                        (global_nn < Nn)
                            ? weight[WEIGHT_IDX(ky, kx, global_nn,
                                                ni_base + ni_off, KX, Nn, Ni)]
                            : 0.f;
                }
                __syncthreads();

                if (valid) {
                    const float *inp =
                        input + INPUT_IDX_3D(b, row + ky, col + kx, ni_base,
                                             NYPAD, NXPAD, Ni);
                    for (int ni = 0; ni < chunk; ni++) {
                        sum += inp[ni] * smem[threadIdx.x * (NI_CHUNK + 1) + ni];
                    }
                }
                __syncthreads();
            }
        }
    }

    if (valid)
        output[OUTPUT_IDX_3D(b, row, col, nn, NYSCL, NXSCL, Nn)] = relu(sum);
}

struct Config {
    int Nx, Ny, Ni, Nn, B;
    const char *name;
};

int main() {
    Config configs[] = {
        {224, 224,  64,  64, 16, "Conv1: 224x224 Ni=Nn=64  B=16"},
        { 14,  14, 512, 512, 16, "Conv2: 14x14   Ni=Nn=512 B=16"},
    };

    for (auto &cfg : configs) {
        int Nx = cfg.Nx, Ny = cfg.Ny, Ni = cfg.Ni, Nn = cfg.Nn, B = cfg.B;
        int NXPAD = Nx + KX - 1;
        int NYPAD = Ny + KY - 1;
        int NXSCL = Nx;  // stride=1
        int NYSCL = Ny;

        size_t input_size  = (size_t)NYPAD * NXPAD * Ni * B * sizeof(float);
        size_t kernel_size = (size_t)KY * KX * Nn * Ni * sizeof(float);
        size_t output_size = (size_t)NYSCL * NXSCL * Nn * B * sizeof(float);

        float *h_input  = (float *)malloc(input_size);
        float *h_weight = (float *)malloc(kernel_size);
        if (!h_input || !h_weight) { std::cerr << "malloc failed\n"; return 1; }

        for (size_t i = 0; i < input_size  / sizeof(float); i++) h_input[i]  = 0.001f * (float)(i % 17 + 1);
        for (size_t i = 0; i < kernel_size / sizeof(float); i++) h_weight[i] = 0.001f * (float)(i % 23 + 1);

        float *d_input, *d_weight, *d_output;
        CHECK_CUDA(cudaMalloc(&d_input,  input_size));
        CHECK_CUDA(cudaMalloc(&d_weight, kernel_size));
        CHECK_CUDA(cudaMalloc(&d_output, output_size));
        CHECK_CUDA(cudaMemcpy(d_input,  h_input,  input_size,  cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_weight, h_weight, kernel_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_output, 0, output_size));

        int Nn_tiles  = (Nn + TILE_N - 1) / TILE_N;
        dim3 block(TILE_N, TILE_SP);
        dim3 grid(NXSCL, (NYSCL + TILE_SP - 1) / TILE_SP, B * Nn_tiles);
        size_t smem_bytes = TILE_N * (NI_CHUNK + 1) * sizeof(float);

        // warmup
        conv2d_smem<<<grid, block, smem_bytes>>>(
            d_weight, d_input, d_output, B, Ny, Nx, Ni, Nn,
            NYPAD, NXPAD, NYSCL, NXSCL);
        CHECK_CUDA(cudaDeviceSynchronize());

        // timed run (ncu intercepts this launch)
        cudaEvent_t t0, t1;
        CHECK_CUDA(cudaEventCreate(&t0));
        CHECK_CUDA(cudaEventCreate(&t1));
        CHECK_CUDA(cudaEventRecord(t0));
        conv2d_smem<<<grid, block, smem_bytes>>>(
            d_weight, d_input, d_output, B, Ny, Nx, Ni, Nn,
            NYPAD, NXPAD, NYSCL, NXSCL);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
        long long ops = 2LL * KY * KX * Ni * B * NYSCL * NXSCL * Nn;
        std::cout << cfg.name << "  " << ms << " ms  "
                  << ops / (ms * 1e-3) / 1e9 << " GFLOPS\n";

        CHECK_CUDA(cudaEventDestroy(t0));
        CHECK_CUDA(cudaEventDestroy(t1));
        CHECK_CUDA(cudaFree(d_input));
        CHECK_CUDA(cudaFree(d_weight));
        CHECK_CUDA(cudaFree(d_output));
        free(h_input);
        free(h_weight);
    }
    return 0;
}
