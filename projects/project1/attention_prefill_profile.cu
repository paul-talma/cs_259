#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define D  64
#define Br 16
#define Bc 16

#define QKV(row, col) ((row)*D + (col))

typedef float VTYPE;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(_e));                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static void fill(VTYPE *m, long long n, int seed) {
    for (long long i = 0; i < n; i++)
        m[i] = 0.1f * sinf((float)(i * 3 + seed * 7));
}

__global__ void prefill_kernel(const VTYPE *__restrict__ Q,
                               const VTYPE *__restrict__ K,
                               const VTYPE *__restrict__ V,
                               VTYPE *__restrict__ O,
                               int S) {

    int d   = threadIdx.x;
    int row = threadIdx.y;
    int i   = blockIdx.x * Br + row;

    __shared__ VTYPE Qi[Br * D];
    __shared__ VTYPE Oi[Br * D];
    __shared__ VTYPE Kj[Bc * D];
    __shared__ VTYPE Vj[Bc * D];
    __shared__ VTYPE Sij[Br * Bc];
    __shared__ VTYPE dot_warp[Br * (D / 32) * Bc];
    __shared__ VTYPE rowmax[Br];
    __shared__ VTYPE rowmaxNew[Br];
    __shared__ VTYPE rsum[Br];
    __shared__ VTYPE rsumNew[Br];
    __shared__ VTYPE corr[Br];

    VTYPE inv_sqrt = 1.f / sqrtf((float)D);

    Qi[row * D + d] = (i < S) ? Q[QKV(i, d)] : 0.f;
    Oi[row * D + d] = 0.f;

    if (d == 0) {
        rowmax[row] = -INFINITY;
        rsum[row]   = 0.f;
    }
    __syncthreads();

    for (int kvOffset = 0; kvOffset < S; kvOffset += Bc) {
        int kvGlobal = kvOffset + row;
        Kj[row * D + d] = (kvGlobal < S) ? K[QKV(kvGlobal, d)] : 0.f;
        Vj[row * D + d] = (kvGlobal < S) ? V[QKV(kvGlobal, d)] : 0.f;
        __syncthreads();

        {
            int warp_in_row = d / 32;
            int lane        = d % 32;
            for (int c = 0; c < Bc; ++c) {
                float partial = Qi[row * D + d] * Kj[c * D + d];
                for (int off = 16; off >= 1; off >>= 1)
                    partial += __shfl_down_sync(0xffffffff, partial, off);
                if (lane == 0)
                    dot_warp[row * (D / 32) * Bc + warp_in_row * Bc + c] = partial;
            }
        }
        __syncthreads();

        if (d < Bc) {
            float acc = 0.f;
            for (int w = 0; w < D / 32; ++w)
                acc += dot_warp[row * (D / 32) * Bc + w * Bc + d];
            int kv_pos = kvOffset + d;
            Sij[row * Bc + d] =
                (i < S && kv_pos <= i) ? acc * inv_sqrt : -INFINITY;
        }
        __syncthreads();

        if (d == 0) {
            float mx = rowmax[row];
            for (int k = 0; k < Bc; ++k)
                mx = fmaxf(mx, Sij[row * Bc + k]);
            rowmaxNew[row] = mx;
            corr[row]      = expf(rowmax[row] - mx);
        }
        __syncthreads();

        if (d < Bc)
            Sij[row * Bc + d] = expf(Sij[row * Bc + d] - rowmaxNew[row]);
        __syncthreads();

        if (d == 0) {
            VTYPE rowSum = 0.f;
            for (int k = 0; k < Bc; ++k)
                rowSum += Sij[row * Bc + k];
            rsumNew[row] = rsum[row] * corr[row] + rowSum;
        }
        __syncthreads();

        VTYPE acc = 0.f;
        for (int k = 0; k < Bc; ++k)
            acc += Sij[row * Bc + k] * Vj[k * D + d];
        Oi[row * D + d] = Oi[row * D + d] * corr[row] + acc;

        if (d == 0) {
            rowmax[row] = rowmaxNew[row];
            rsum[row]   = rsumNew[row];
        }
        __syncthreads();
    }

    if (i < S)
        O[QKV(i, d)] = Oi[row * D + d] / rsum[row];
}

static void run(int S) {
    long long n     = (long long)S * D;
    long long flops = 2LL * S * (S + 1) * D;

    VTYPE *Q = (VTYPE *)malloc(n * sizeof(VTYPE));
    VTYPE *K = (VTYPE *)malloc(n * sizeof(VTYPE));
    VTYPE *V = (VTYPE *)malloc(n * sizeof(VTYPE));
    if (!Q || !K || !V) { fprintf(stderr, "malloc failed\n"); exit(1); }

    fill(Q, n, 1);
    fill(K, n, 2);
    fill(V, n, 3);

    VTYPE *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, n * sizeof(VTYPE)));
    CUDA_CHECK(cudaMalloc(&d_K, n * sizeof(VTYPE)));
    CUDA_CHECK(cudaMalloc(&d_V, n * sizeof(VTYPE)));
    CUDA_CHECK(cudaMalloc(&d_O, n * sizeof(VTYPE)));
    CUDA_CHECK(cudaMemcpy(d_Q, Q, n * sizeof(VTYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, K, n * sizeof(VTYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, V, n * sizeof(VTYPE), cudaMemcpyHostToDevice));

    dim3 block(D, Br);
    dim3 grid((S + Br - 1) / Br, 1);

    // warmup
    prefill_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, S);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    prefill_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    printf("PREFILL S=%-6d  %.4f ms  %.4f GFLOPS\n",
           S, ms, flops * 1e-9 / (ms * 1e-3));

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    free(Q); free(K); free(V);
}

int main(void) {
    run(4096);
    run(65536);
    return 0;
}
