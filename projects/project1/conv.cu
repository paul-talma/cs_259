/*
 * conv.cu -- CUDA implementation of direct 2D convolution.
 *
 * VGG configurations:
 *   Conv1: 224x224  3x3  Ni=64   Nn=64
 *   Conv2:  14x14   3x3  Ni=512  Nn=512
 *
 * Data layout (matches CPU reference):
 *   Weights: [KY][KX][Nn][Ni]
 *   Input:   [B][NYPAD][NXPAD][Ni]   (zero-padded by one pixel on each side)
 *   Output:  [B][NYSCL][NXSCL][Nn]
 *
 * Compile: make
 * Run:     ./conv
 */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

#define KY 3 /* kernel height  */
#define KX 3 /* kernel width   */
#define SY 1 /* stride y       */
#define SX 1 /* stride x       */

typedef float VTYPE;

/* ------------------------------------------------------------------ */
/* CUDA error checking                                                 */
/* ------------------------------------------------------------------ */

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_e));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define INPUT_IDX(row, col, inChannel, inWidth, inChannels)                    \
    (((row) * (inWidth) * (inChannels)) + ((col) * (inChannels)) + (inChannel))

#define FILTER_IDX(row, col, outChannel, inChannel, filterWidth, outChannels,  \
                   inChannels)                                                 \
    (((row) * (filterWidth) * (outChannels) * (inChannels)) +                  \
     ((col) * (outChannels) * (inChannels)) + ((outChannel) * (inChannels)) +  \
     (inChannel))

#define OUTPUT_IDX(row, col, outChannel, outWidth, outChannels)                \
    (((row) * (outWidth) * (outChannels)) + ((col) * (outChannels)) +          \
     (outChannel))
/* ------------------------------------------------------------------ */
/* Helpers shared with CPU reference                                   */
/* ------------------------------------------------------------------ */

static __host__ __device__ VTYPE relu(VTYPE x) { return x > 0.f ? x : 0.f; }

/* Deterministic fill — must match the CPU reference exactly. */
static void fill(VTYPE *m, long long n, float scale, int seed) {
    for (long long i = 0; i < n; i++)
        m[i] = scale * sinf((float)(i * 3 + seed * 7));
}

/* ------------------------------------------------------------------ */
/* CPU reference (for correctness verification)                        */
/* ------------------------------------------------------------------ */

static void
cpu_convolution_layer(VTYPE *synapse,  /* [KY * KX * Nn * Ni]      */
                      VTYPE *neuron_i, /* [B * NYPAD * NXPAD * Ni] */
                      VTYPE *neuron_n, /* [B * NYSCL * NXSCL * Nn] */
                      int B,
                      int Ny,
                      int Nx,
                      int Ni,
                      int Nn) {
    int NXPAD = Nx + KX - 1, NYPAD = Ny + KY - 1;
    int NXSCL = (Nx + SX - 1) / SX, NYSCL = (Ny + SY - 1) / SY;

#pragma omp parallel for schedule(static) collapse(3)
    for (int b = 0; b < B; b++)
        for (int y = 0; y < Ny; y += SY)
            for (int x = 0; x < Nx; x += SX) {
                int yout = y / SY, xout = x / SX;
                for (int nn = 0; nn < Nn; nn++) {
                    VTYPE sum = 0.f;
                    for (int ky = 0; ky < KY; ky++)
                        for (int kx = 0; kx < KX; kx++)
                            for (int i = 0; i < Ni; i++)
                                sum += synapse[FILTER_IDX(ky, kx, nn, i, KX, Nn, Ni)] *
                                       neuron_i[INPUT_IDX(b * NYPAD + ky + y, kx + x, i, NXPAD, Ni)];
                    neuron_n[OUTPUT_IDX(b * NYSCL + yout, xout, nn, NXSCL, Nn)] = relu(sum);
                }
            }
}

/* ------------------------------------------------------------------ */
/* CUDA kernel (stub — implement this)                                 */
/* ------------------------------------------------------------------ */

__global__ void
conv_kernel(const VTYPE *__restrict__ synapse,  /* [KY][KX][Nn][Ni] */
            const VTYPE *__restrict__ neuron_i, /* [B][NYPAD][NXPAD][Ni] */
            VTYPE *__restrict__ neuron_n,       /* [B][NYSCL][NXSCL][Nn] */
            int B,
            int Ny,
            int Nx,
            int Ni,
            int Nn) {
    // padded input size
    int NXPAD = Nx + KX - 1, NYPAD = Ny + KY - 1;
    // output size
    int NXSCL = (Nx + SX - 1) / SX, NYSCL = (Ny + SY - 1) / SY;

    size_t outputCol = blockDim.x * blockIdx.x + threadIdx.x;
    size_t outputRow = blockDim.y * blockIdx.y + threadIdx.y;
    int b = blockIdx.z;

    if (outputCol >= (size_t)NXSCL || outputRow >= (size_t)NYSCL)
        return;

    for (int nn = 0; nn < Nn; nn++) {
        float sum = 0.f;
        for (int ky = 0; ky < KY; ky++)
            for (int kx = 0; kx < KX; kx++)
                for (int i = 0; i < Ni; i++)
                    sum += synapse[FILTER_IDX(ky, kx, nn, i, KX, Nn, Ni)] *
                           neuron_i[(size_t)b * NYPAD * NXPAD * Ni +
                                    INPUT_IDX(outputRow + ky, outputCol + kx, i, NXPAD, Ni)];

        neuron_n[(size_t)b * NYSCL * NXSCL * Nn +
                 OUTPUT_IDX(outputRow, outputCol, nn, NXSCL, Nn)] = relu(sum);
    }
}

/* ------------------------------------------------------------------ */
/* Verify GPU output against CPU reference                             */
/* ------------------------------------------------------------------ */
static bool verify(const VTYPE *gpu_out,
                   const VTYPE *cpu_out,
                   long long n,
                   float tol = 1e-3f) {
    long long mismatches = 0;
    for (long long i = 0; i < n; i++) {
        float diff = fabsf(gpu_out[i] - cpu_out[i]);
        float ref = fabsf(cpu_out[i]) + 1e-6f;
        if (diff / ref > tol) {
            if (mismatches < 5)
                fprintf(stderr, "  mismatch at [%lld]: gpu=%.6f  cpu=%.6f\n", i,
                        gpu_out[i], cpu_out[i]);
            mismatches++;
        }
    }
    if (mismatches > 0)
        fprintf(stderr, "  %lld / %lld elements exceed tolerance %.1e\n",
                mismatches, n, tol);
    return mismatches == 0;
}

/* ------------------------------------------------------------------ */
/* Run one configuration                                               */
/* ------------------------------------------------------------------ */

static void run(const char *name, int B, int Ny, int Nx, int Ni, int Nn) {
    int NYPAD = Ny + KY - 1, NXPAD = Nx + KX - 1;
    int NYSCL = (Ny + SY - 1) / SY, NXSCL = (Nx + SX - 1) / SX;

    long long syn_n = (long long)KY * KX * Nn * Ni;
    long long inp_n = (long long)B * NYPAD * NXPAD * Ni;
    long long out_n = (long long)B * NYSCL * NXSCL * Nn;
    long long flops = 2LL * B * NYSCL * NXSCL * KY * KX * Ni * Nn;

    /* ---- host buffers -------------------------------------------- */
    VTYPE *h_syn = (VTYPE *)malloc(syn_n * sizeof(VTYPE));
    VTYPE *h_inp =
        (VTYPE *)calloc(inp_n, sizeof(VTYPE)); /* calloc = zero padding */
    VTYPE *h_out_cpu = (VTYPE *)malloc(out_n * sizeof(VTYPE));
    VTYPE *h_out_gpu = (VTYPE *)malloc(out_n * sizeof(VTYPE));

    /* ---- initialise weights (same seed as CPU reference) ---------- */
    fill(h_syn, syn_n, 0.01f, 1);

    /* ---- initialise input (interior pixels only; border stays 0) -- */
    for (int b = 0; b < B; b++)
        for (int y = 0; y < Ny; y++)
            for (int x = 0; x < Nx; x++)
                for (int i = 0; i < Ni; i++)
                    h_inp[((long long)(b * NYPAD + y) * NXPAD + x) * Ni + i] =
                        0.01f * sinf((float)(b * Ny * Nx * Ni + y * Nx * Ni +
                                             x * Ni + i));

    /* ---- CPU reference for correctness ---------------------------- */
    cpu_convolution_layer(h_syn, h_inp, h_out_cpu, B, Ny, Nx, Ni, Nn);

    /* ---- GPU buffers ---------------------------------------------- */
    VTYPE *d_syn, *d_inp, *d_out;
    CUDA_CHECK(cudaMalloc(&d_syn, syn_n * sizeof(VTYPE)));
    CUDA_CHECK(cudaMalloc(&d_inp, inp_n * sizeof(VTYPE)));
    CUDA_CHECK(cudaMalloc(&d_out, out_n * sizeof(VTYPE)));

    CUDA_CHECK(cudaMemcpy(d_syn, h_syn, syn_n * sizeof(VTYPE),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inp, h_inp, inp_n * sizeof(VTYPE),
                          cudaMemcpyHostToDevice));

    /* ---- kernel launch -------------------------------------------- */
    /* TODO: tune grid/block dimensions for your kernel */
    dim3 block(16, 16, 1);
    dim3 grid((NXSCL + block.x - 1) / block.x, (NYSCL + block.y - 1) / block.y,
              B);

    /* warm-up */
    conv_kernel<<<grid, block>>>(d_syn, d_inp, d_out, B, Ny, Nx, Ni, Nn);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* timed run */
    cudaEvent_t t_start, t_stop;
    CUDA_CHECK(cudaEventCreate(&t_start));
    CUDA_CHECK(cudaEventCreate(&t_stop));

    CUDA_CHECK(cudaEventRecord(t_start));
    conv_kernel<<<grid, block>>>(d_syn, d_inp, d_out, B, Ny, Nx, Ni, Nn);
    CUDA_CHECK(cudaEventRecord(t_stop));
    CUDA_CHECK(cudaEventSynchronize(t_stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t_start, t_stop));
    CUDA_CHECK(cudaEventDestroy(t_start));
    CUDA_CHECK(cudaEventDestroy(t_stop));

    /* ---- copy result back and verify ------------------------------ */
    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, out_n * sizeof(VTYPE),
                          cudaMemcpyDeviceToHost));

    bool ok = verify(h_out_gpu, h_out_cpu, out_n);
    double gflops = (double)flops * 1e-9 / (ms * 1e-3);

    printf("  %-12s B=%-2d  %8.2f ms  %8.2f GFLOPS  %s\n", name, B, ms, gflops,
           ok ? "PASS" : "FAIL");

    /* ---- cleanup -------------------------------------------------- */
    CUDA_CHECK(cudaFree(d_syn));
    CUDA_CHECK(cudaFree(d_inp));
    CUDA_CHECK(cudaFree(d_out));
    free(h_syn);
    free(h_inp);
    free(h_out_cpu);
    free(h_out_gpu);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(void) {
    int dev;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("=== Convolution CUDA  device: %s ===\n\n", prop.name);

    printf("  %-18s  %10s  %10s  %s\n", "Layer", "ms", "GFLOPS", "Correct");
    printf("  %s\n",
           "─────────────────────────────────────────────────────────────");

    /*        name     B    Ny   Nx   Ni    Nn  */
    run("Conv1-VGG", 1, 224, 224, 64, 64);
    run("Conv1-VGG", 16, 224, 224, 64, 64);
    run("Conv2-VGG", 1, 14, 14, 512, 512);
    run("Conv2-VGG", 16, 14, 14, 512, 512);

    printf("\n");
    return 0;
}
