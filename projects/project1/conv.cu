#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef float VTYPE;
// Convolution parameters
#define KY 3
#define KX 3
#define TILE_N                                                                 \
    32 // output channels per block; equals warp size for full coalescing
#define TILE_SP 8   // spatial rows per block (for smem weight reuse)
#define NI_CHUNK 64 // input channels loaded into smem per iteration

// Macros for indexing
#define INPUT_IDX_3D(b, y, x, ni, NYPAD, NXPAD, Ni)                            \
    ((((y) * (NXPAD) + (x)) * (Ni)) + (ni) + (b) * (NYPAD) * (NXPAD) * (Ni))

#define OUTPUT_IDX_3D(b, y, x, nn, NYSCL, NXSCL, Nn)                           \
    ((((y) * (NXSCL) + (x)) * (Nn)) + (nn) + (b) * (NYSCL) * (NXSCL) * (Nn))

#define SHARED_WEIGHT_IDX(ky, kx, ni, KX, Ni)                                  \
    (((ky) * (KX) + (kx)) * (Ni) + (ni))

#define WEIGHT_IDX(ky, kx, nn, ni, KX, Nn, Ni)                                 \
    ((((ky) * (KX) + (kx)) * (Nn) + (nn)) * (Ni) + (ni))

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " -> " << cudaGetErrorString(err__) << std::endl;     \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

__device__ float relu(float x) { return (x > 0.0f) ? x : 0.0f; }

float relu_cpu(float x) { return (x > 0.0f) ? x : 0.0f; }

// GPU kernel
__global__ void conv2d(const float *weight,
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

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int bz = blockIdx.z;
    int nn = bz % Nn; // feature map index
    int b = bz / Nn;  // batch index

    float output_entry = 0.0f;
    float input_entry;
    float weight_entry;

    if (col < Nx && row < Ny) {
        for (int ky = 0; ky < KY; ky++) {
            for (int kx = 0; kx < KX; kx++) {
                for (int ni = 0; ni < Ni; ni++) {
                    input_entry = input[INPUT_IDX_3D(b, ky + row, kx + col, ni,
                                                     NYPAD, NXPAD, Ni)];
                    weight_entry =
                        weight[WEIGHT_IDX(ky, kx, nn, ni, KX, Nn, Ni)];
                    // shared_weights[SHARED_WEIGHT_IDX(ky, kx, ni, KX, Ni)];
                    output_entry += input_entry * weight_entry;
                }
            }
        }

        output_entry = relu(output_entry);
        output[OUTPUT_IDX_3D(b, row, col, nn, NYSCL, NXSCL, Nn)] = output_entry;
    }
}

// Optimized kernel: threadIdx.x indexes nn (output channel) so the 32 warp
// lanes write 32 consecutive channel values — fully coalesced output store.
// Input is broadcast across the warp (all lanes share the same spatial point).
__global__ void conv2d_optimized(const float *weight,
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

    int col = blockIdx.x;
    int row = blockIdx.y;
    int Nn_tiles = (Nn + TILE_N - 1) / TILE_N;
    int nn = (blockIdx.z % Nn_tiles) * TILE_N + threadIdx.x;
    int b = blockIdx.z / Nn_tiles;

    if (nn >= Nn)
        return;

    float sum = 0.f;
    for (int ky = 0; ky < KY; ky++) {
        for (int kx = 0; kx < KX; kx++) {
            // All warp lanes share inp (same spatial point) -> broadcast, 1 txn
            const float *inp = input + INPUT_IDX_3D(b, row + ky, col + kx, 0,
                                                    NYPAD, NXPAD, Ni);
            // Each lane has its own nn -> w is strided by Ni across lanes,
            // but the TILE_N*Ni slice is contiguous and fits in L2 cache.
            const float *w = weight + WEIGHT_IDX(ky, kx, nn, 0, KX, Nn, Ni);
            for (int ni = 0; ni < Ni; ni++) {
                sum += inp[ni] * w[ni];
            }
        }
    }

    // Coalesced: 32 consecutive nn -> 32 consecutive floats in one cache line
    output[OUTPUT_IDX_3D(b, row, col, nn, NYSCL, NXSCL, Nn)] = relu(sum);
}

// Smem kernel: combines channel coalescing (threadIdx.x -> nn) with spatial
// tiling (TILE_SP rows per block) so TILE_SP threads share each weight load.
// Per (ky, kx, ni_chunk): all 256 threads cooperatively load a
// [TILE_N][NI_CHUNK] weight slice into smem, then each of the TILE_SP spatial
// rows computes its partial dot-product using those cached weights.
// smem layout: [TILE_N][NI_CHUNK+1] — the +1 padding column ensures that for
// any fixed ni, thread t reads bank (t + ni) % 32, giving all 32 banks.
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

    extern __shared__ float smem[]; // TILE_N * (NI_CHUNK + 1) floats

    int col = blockIdx.x;
    int row = blockIdx.y * TILE_SP + threadIdx.y;
    int Nn_tiles = (Nn + TILE_N - 1) / TILE_N;
    int nn_tile = blockIdx.z % Nn_tiles;
    int b = blockIdx.z / Nn_tiles;
    int nn = nn_tile * TILE_N + threadIdx.x;
    bool valid = (nn < Nn) && (row < NYSCL);

    int tid = threadIdx.y * TILE_N + threadIdx.x;
    int block_threads = TILE_N * TILE_SP;

    float sum = 0.f;

    for (int ky = 0; ky < KY; ky++) {
        for (int kx = 0; kx < KX; kx++) {
            for (int ni_base = 0; ni_base < Ni; ni_base += NI_CHUNK) {
                int chunk =
                    (ni_base + NI_CHUNK <= Ni) ? NI_CHUNK : (Ni - ni_base);

                // Cooperatively load weight[ky][kx][nn_tile*TILE_N..+TILE_N)
                //                                  [ni_base..+chunk) into smem.
                // Within each nn row, NI_CHUNK values are contiguous in global
                // memory, so each warp's load touches consecutive addresses.
                for (int s = tid; s < TILE_N * chunk; s += block_threads) {
                    int nn_off = s / chunk;
                    int ni_off = s % chunk;
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
                        // bank = (threadIdx.x * (NI_CHUNK+1) + ni) % 32
                        //      = (threadIdx.x + ni) % 32  -> all 32 banks
                        sum +=
                            inp[ni] * smem[threadIdx.x * (NI_CHUNK + 1) + ni];
                    }
                }
                __syncthreads();
            }
        }
    }

    if (valid)
        output[OUTPUT_IDX_3D(b, row, col, nn, NYSCL, NXSCL, Nn)] = relu(sum);
}

// CPU implementation
void conv2d_cpu(const float *weight,
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
#pragma omp parallel for collapse(4) schedule(static)
    for (int b = 0; b < B; b++) {
        for (int nn = 0; nn < Nn; nn++) {
            for (int row = 0; row < Ny; row++) {
                for (int col = 0; col < Nx; col++) {
                    float sum = 0.0f;
                    for (int ky = 0; ky < KY; ky++) {
                        for (int kx = 0; kx < KX; kx++) {
                            for (int ni = 0; ni < Ni; ni++) {
                                float inp =
                                    input[INPUT_IDX_3D(b, row + ky, col + kx,
                                                       ni, NYPAD, NXPAD, Ni)];
                                float w = weight[WEIGHT_IDX(ky, kx, nn, ni, KX,
                                                            Nn, Ni)];
                                sum += inp * w;
                            }
                        }
                    }
                    output[OUTPUT_IDX_3D(b, row, col, nn, NYSCL, NXSCL, Nn)] =
                        relu_cpu(sum);
                }
            }
        }
    }
}

struct Config {
    int Nx, Ny, Ni, Nn, B, stride;
    std::string name;
};

int main() {
    std::vector<Config> configs = {
        {224, 224, 64, 64, 1, 1, "Conv1: Nx=Ny=224, Kx=Ky=3, Ni=Nn=64, B=1"},
        {224, 224, 64, 64, 16, 1, "Conv1: Nx=Ny=224, Kx=Ky=3, Ni=Nn=64, B=16"},
        {14, 14, 512, 512, 1, 1, "Conv2: Nx=Ny=14, Kx=Ky=3, Ni=Nn=512, B=1"},
        {14, 14, 512, 512, 16, 1, "Conv2: Nx=Ny=14, Kx=Ky=3, Ni=Nn=512, B=16"},
    };

    std::cout << std::fixed << std::setprecision(6);
#ifdef _OPENMP
    std::cout << "OpenMP CPU threads: " << omp_get_max_threads() << std::endl
              << std::endl;
#else
    std::cout << "OpenMP CPU threads: disabled (compile with "
                 "-Xcompiler -fopenmp)"
              << std::endl
              << std::endl;
#endif

    for (auto &cfg : configs) {
        int Nx = cfg.Nx, Ny = cfg.Ny, Ni = cfg.Ni, Nn = cfg.Nn, B = cfg.B;
        int NXPAD = Nx + KX - 1;
        int NYPAD = Ny + KY - 1;
        int NXSCL = (Nx + cfg.stride - 1) / cfg.stride;
        int NYSCL = (Ny + cfg.stride - 1) / cfg.stride;

        size_t input_size = (size_t)NYPAD * NXPAD * Ni * B * sizeof(float);
        size_t kernel_size = (size_t)KY * KX * Nn * Ni * sizeof(float);
        size_t output_size = (size_t)NYSCL * NXSCL * Nn * B * sizeof(float);

        // Allocate host memory
        float *h_input       = (float *)malloc(input_size);
        float *h_weight      = (float *)malloc(kernel_size);
        float *h_output_cpu  = (float *)malloc(output_size);
        float *h_output_smem = (float *)malloc(output_size);

        if (!h_input || !h_weight || !h_output_cpu || !h_output_smem) {
            std::cerr << "Host allocation failed for " << cfg.name << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Initialize input with zero padding and varied positive values inside.
        for (int b = 0; b < B; b++) {
            for (int y = 0; y < NYPAD; y++) {
                for (int x = 0; x < NXPAD; x++) {
                    for (int ni = 0; ni < Ni; ni++) {
                        bool is_padding = (y < KY / 2 || y >= NYPAD - KY / 2 ||
                                           x < KX / 2 || x >= NXPAD - KX / 2);
                        float value = 0.001f * (float)((b + 1) + (y % 17) +
                                                       (x % 13) + (ni % 11));
                        h_input[INPUT_IDX_3D(b, y, x, ni, NYPAD, NXPAD, Ni)] =
                            is_padding ? 0.0f : value;
                    }
                }
            }
        }

        // Initialize weights with varied positive values so
        // channel/filter indexing bugs show up.
        for (size_t i = 0; i < kernel_size / sizeof(float); i++) {
            h_weight[i] = 0.001f * (float)((i % 23) + 1);
        }

        // CPU reference (for correctness check only)
        conv2d_cpu(h_weight, h_input, h_output_cpu, B, Ny, Nx, Ni, Nn, NYPAD,
                   NXPAD, NYSCL, NXSCL);

        // GPU memory
        float *d_input, *d_weight;
        CHECK_CUDA(cudaMalloc((void **)&d_input, input_size));
        CHECK_CUDA(cudaMalloc((void **)&d_weight, kernel_size));
        CHECK_CUDA(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_weight, h_weight, kernel_size, cudaMemcpyHostToDevice));

        // ---- conv2d_smem ----
        float *d_output_smem;
        CHECK_CUDA(cudaMalloc((void **)&d_output_smem, output_size));
        CHECK_CUDA(cudaMemset(d_output_smem, 0, output_size));

        int Nn_tiles = (Nn + TILE_N - 1) / TILE_N;
        dim3 block_smem(TILE_N, TILE_SP);
        dim3 grid_smem(NXSCL, (NYSCL + TILE_SP - 1) / TILE_SP, B * Nn_tiles);
        size_t smem_bytes = TILE_N * (NI_CHUNK + 1) * sizeof(float);

        // warmup
        conv2d_smem<<<grid_smem, block_smem, smem_bytes>>>(
            d_weight, d_input, d_output_smem, B, Ny, Nx, Ni, Nn,
            NYPAD, NXPAD, NYSCL, NXSCL);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start_smem, stop_smem;
        CHECK_CUDA(cudaEventCreate(&start_smem));
        CHECK_CUDA(cudaEventCreate(&stop_smem));
        CHECK_CUDA(cudaEventRecord(start_smem));
        conv2d_smem<<<grid_smem, block_smem, smem_bytes>>>(
            d_weight, d_input, d_output_smem, B, Ny, Nx, Ni, Nn,
            NYPAD, NXPAD, NYSCL, NXSCL);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop_smem));
        CHECK_CUDA(cudaEventSynchronize(stop_smem));

        float smem_time_ms;
        CHECK_CUDA(cudaEventElapsedTime(&smem_time_ms, start_smem, stop_smem));

        CHECK_CUDA(cudaMemcpy(h_output_smem, d_output_smem, output_size,
                              cudaMemcpyDeviceToHost));

        bool correct_smem  = true;
        float max_diff_smem = 0.0f;
        for (size_t i = 0; i < output_size / sizeof(float); i++) {
            float diff = std::fabs(h_output_cpu[i] - h_output_smem[i]);
            if (diff > max_diff_smem) max_diff_smem = diff;
            if (diff > 1e-5f) correct_smem = false;
        }

        long long ops = 2LL * KY * KX * Ni * B * NYSCL * NXSCL * Nn;
        double smem_gflops = ops / (smem_time_ms * 1e-3) / 1e9;

        std::cout << cfg.name << std::endl;
        std::cout << "  conv2d_smem: " << smem_time_ms << " ms  "
                  << smem_gflops << " GFLOPS  "
                  << (correct_smem ? "PASS" : "FAIL")
                  << " (max diff: " << max_diff_smem << ")" << std::endl;
        std::cout << std::endl;

        // Cleanup
        free(h_input);
        free(h_weight);
        free(h_output_cpu);
        free(h_output_smem);
        CHECK_CUDA(cudaFree(d_input));
        CHECK_CUDA(cudaFree(d_weight));
        CHECK_CUDA(cudaFree(d_output_smem));
        CHECK_CUDA(cudaEventDestroy(start_smem));
        CHECK_CUDA(cudaEventDestroy(stop_smem));
    }

    return 0;
}
