/*
 * attention.cu -- CUDA implementation of single-head causal attention.
 *
 * Two GPU kernel stubs to implement and optimise:
 *   prefill_kernel -- S queries attending causally to S keys/values
 *   decode_kernel  -- single query attending to a KV cache of size C
 *
 * Data layout (flat row-major; matches CPU reference strides):
 *   Prefill  Q, K, V, O : float[S * D]
 *   Decode   q, o       : float[D]
 *            K, V       : float[C * D]
 *
 *   QKV(row, col) = row * D + col
 *
 * FLOP accounting:
 *   Prefill (causal triangle): 2 * S*(S+1) * D   (QK pass + AV pass)
 *   Decode:                    4 * C * D          (QK pass + AV pass)
 *
 * Compile:  make attention
 * Run:      ./attention
 */

#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

#define D 64 /* head dimension — compile-time constant */

/* Row-major indexing into a [rows][D] matrix stored as a flat array */
#define QKV(row, col) ((row)*D + (col))

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

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

/* Deterministic fill — same formula as the CPU reference. */
static void fill(VTYPE *m, long long n, int seed) {
    for (long long i = 0; i < n; i++)
        m[i] = 0.1f * sinf((float)(i * 3 + seed * 7));
}

/* ------------------------------------------------------------------ */
/* CPU reference implementations                                       */
/* ------------------------------------------------------------------ */

/*
 * standard_prefill -- two-pass softmax.
 *
 * For each query i: compute all QK scores, find max, exponentiate and
 * normalise, then accumulate weighted values.  Allocates one score
 * buffer of size S per OpenMP thread.  Skip for large S (O(S^2) cost).
 */
static void standard_prefill(const VTYPE *Q, /* [S * D] */
                             const VTYPE *K, /* [S * D] */
                             const VTYPE *V, /* [S * D] */
                             VTYPE *O,       /* [S * D] */
                             int S) {
    float inv_sqrt = 1.f / sqrtf((float)D);
    int nthreads = omp_get_max_threads();
    VTYPE *score_bufs =
        (VTYPE *)malloc((long long)nthreads * S * sizeof(VTYPE));

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < S; i++) {
        VTYPE *scores = score_bufs + (long long)omp_get_thread_num() * S;

        /* QK dot products */
        for (int j = 0; j <= i; j++) {
            float s = 0.f;
            for (int d = 0; d < D; d++)
                s += Q[QKV(i, d)] * K[QKV(j, d)];
            scores[j] = s * inv_sqrt;
        }

        /* softmax */
        float mx = scores[0];
        for (int j = 1; j <= i; j++)
            if (scores[j] > mx)
                mx = scores[j];
        float sum = 0.f;
        for (int j = 0; j <= i; j++) {
            scores[j] = expf(scores[j] - mx);
            sum += scores[j];
        }
        for (int j = 0; j <= i; j++)
            scores[j] /= sum;

        /* weighted value accumulation */
        for (int d = 0; d < D; d++) {
            float acc = 0.f;
            for (int j = 0; j <= i; j++)
                acc += scores[j] * V[QKV(j, d)];
            O[QKV(i, d)] = acc;
        }
    }
    free(score_bufs);
}

/*
 * flash_prefill -- online softmax, single pass, no O(S) scores buffer.
 *
 * Streams keys and values one at a time, maintaining a running max m,
 * normaliser den, and output accumulator out[D].  When a new score
 * exceeds m, previously accumulated values are rescaled by
 * exp(m_old - m_new) before the new contribution is added.
 *
 * Use this as the template for prefill_kernel.
 */
static void flash_prefill(const VTYPE *Q, /* [S * D] */
                          const VTYPE *K, /* [S * D] */
                          const VTYPE *V, /* [S * D] */
                          VTYPE *O,       /* [S * D] */
                          int S) {
    float inv_sqrt = 1.f / sqrtf((float)D);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < S; i++) {
        float out[D]; /* D=64: fixed-size, lives on the stack */
        for (int d = 0; d < D; d++)
            out[d] = 0.f;
        float m = -FLT_MAX, den = 0.f;

        for (int j = 0; j <= i; j++) {
            float score = 0.f;
            for (int d = 0; d < D; d++)
                score += Q[QKV(i, d)] * K[QKV(j, d)];
            score *= inv_sqrt;

            float m_new = (score > m) ? score : m;
            float correction =
                expf(m - m_new); /* rescale factor for old state */
            float exp_score = expf(score - m_new);

            den = den * correction + exp_score;
            for (int d = 0; d < D; d++)
                out[d] = out[d] * correction + exp_score * V[QKV(j, d)];

            m = m_new;
        }

        for (int d = 0; d < D; d++)
            O[QKV(i, d)] = out[d] / den;
    }
}

/*
 * standard_decode -- single query attends to all C entries in the KV cache.
 */
static void standard_decode(const VTYPE *q, /* [D]     */
                            const VTYPE *K, /* [C * D] */
                            const VTYPE *V, /* [C * D] */
                            VTYPE *o,       /* [D]     */
                            int C) {
    float inv_sqrt = 1.f / sqrtf((float)D);
    VTYPE *scores = (VTYPE *)malloc(C * sizeof(VTYPE));

    for (int j = 0; j < C; j++) {
        float s = 0.f;
        for (int d = 0; d < D; d++)
            s += q[d] * K[QKV(j, d)];
        scores[j] = s * inv_sqrt;
    }

    float mx = scores[0];
    for (int j = 1; j < C; j++)
        if (scores[j] > mx)
            mx = scores[j];
    float sum = 0.f;
    for (int j = 0; j < C; j++) {
        scores[j] = expf(scores[j] - mx);
        sum += scores[j];
    }
    for (int j = 0; j < C; j++)
        scores[j] /= sum;

    for (int d = 0; d < D; d++) {
        float acc = 0.f;
        for (int j = 0; j < C; j++)
            acc += scores[j] * V[QKV(j, d)];
        o[d] = acc;
    }
    free(scores);
}

/* ------------------------------------------------------------------ */
/* CUDA kernel stubs — implement and optimise these                    */
/* ------------------------------------------------------------------ */

/*
 * standard_prefill_kernel -- two-pass softmax, one thread per (query,
 * output-dim) pair.
 *
 * Parallelisation: Thread (d, i) computes O[i][d].
 *
 * Each thread allocates a private scores[S] buffer in global memory (passed in
 * as a pre-allocated workspace) or in shared memory for smaller S.  It then
 * runs the three passes: QK dot-products, softmax, weighted-value sum.
 *
 * Limitations / optimisation targets:
 *  - scores[] workspace is S floats per query — O(S^2) total; infeasible for
 *    S=65536 without tiling.
 *  - All threads for the same query redundantly compute the same QK scores;
 *    compute scores once per query (one thread or a cooperative block) and
 *    share via shared memory.
 *  - The softmax max/sum reduction across j requires inter-thread coordination
 *    (atomics, warp shuffle, or separate reduction kernel).
 */
__global__ void standard_prefill_kernel(
    const VTYPE *__restrict__ Q, /* [S * D] */
    const VTYPE *__restrict__ K, /* [S * D] */
    const VTYPE *__restrict__ V, /* [S * D] */
    VTYPE *__restrict__ O,       /* [S * D] */
    VTYPE *__restrict__ scores,  /* [S * S] workspace — caller allocates */
    int S) {

    // S blocks, each of D threads
    // thread (i, d) computes O[i,d]
    int i = blockIdx.x;
    int d = threadIdx.x;

    VTYPE *local_scores = scores + S * i;

    float inv_sqrt = 1.f / sqrtf((float)D);

    if (d == 0) {
        // compute scores for query i
        for (int j = 0; j <= i; ++j) {
            // compute scores[i, j]
            float score = 0;
            // dot product
            for (int k = 0; k < D; ++k) {
                score += Q[QKV(i, k)] * K[QKV(j, k)];
            }
            score *= inv_sqrt;
            local_scores[j] = score;
        }

        // get max
        float mx = local_scores[0];
        for (int j = 0; j <= i; ++j) {
            if (local_scores[j] > mx) {
                mx = local_scores[j];
            }
        }

        // normalize
        float sum = 0.f;
        for (int j = 0; j <= i; ++j) {
            local_scores[j] = expf(local_scores[j] - mx);
            sum += local_scores[j];
        }

        for (int j = 0; j <= i; ++j) {
            local_scores[j] /= sum;
        }
    }
    __syncthreads();

    // combine values
    float acc = 0.f;
    for (int j = 0; j <= i; ++j) {
        acc += local_scores[j] * V[QKV(j, d)];
    }
    O[QKV(i, d)] = acc;
}

/*
 * prefill_kernel -- flash attention, one thread per query row.
 *
 * Parallelisation: a 1-D grid of S threads, one thread per query i.
 * Each thread independently runs flash attention over keys j = 0..i,
 * accumulating into a private out[D] register array (D=64 floats).
 *
 * Limitations / optimisation targets:
 *  - No intra-query parallelism: threads do not cooperate on the D-wide
 *    dot products or value accumulations.
 *  - Register pressure: out[D] costs 64 registers per thread, which
 *    limits the number of resident warps.  Consider mapping a thread
 *    block to each query so threads share the work across D.
 *  - Memory traffic: Q[i] and every K[j] / V[j] row is read from DRAM
 *    on every query.  Tile K and V into shared memory to amortise.
 *  - Warp-level dot products: use __shfl_down_sync to reduce the D
 *    partial products within a warp rather than looping serially.
 */
__global__ void prefill_kernel(const VTYPE *__restrict__ Q, /* [S * D] */
                               const VTYPE *__restrict__ K, /* [S * D] */
                               const VTYPE *__restrict__ V, /* [S * D] */
                               VTYPE *__restrict__ O,       /* [S * D] */
                               int S) {
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if (i >= S)
    //     return;
    //
    // float inv_sqrt = 1.f / sqrtf((float)D);
    //
    // /* Private accumulator — D=64 floats live in registers. */
    // float out[D];
    // for (int d = 0; d < D; d++)
    //     out[d] = 0.f;
    // float m = -FLT_MAX, den = 0.f;
    //
    // /* Causal: query i attends to keys j = 0..i */
    // for (int j = 0; j <= i; j++) {
    //     float score = 0.f;
    //     for (int d = 0; d < D; d++)
    //         score += Q[QKV(i, d)] * K[QKV(j, d)];
    //     score *= inv_sqrt;
    //
    //     float m_new = (score > m) ? score : m;
    //     float correction = expf(m - m_new);
    //     float exp_score = expf(score - m_new);
    //
    //     den = den * correction + exp_score;
    //     for (int d = 0; d < D; d++)
    //         out[d] = out[d] * correction + exp_score * V[QKV(j, d)];
    //
    //     m = m_new;
    // }
    //
    // for (int d = 0; d < D; d++)
    //     O[QKV(i, d)] = out[d] / den;
}

/*
 * decode_kernel -- flash attention for one query, one thread per output
 * dim.
 *
 * Parallelisation: a single block of D=64 threads.  Thread d owns
 * output element o[d] and streams through the KV cache applying online
 * softmax.  All D threads compute the same QK dot product independently
 * (D-fold redundancy) so the running state (m, den) remains consistent
 * across threads without synchronisation.
 *
 * Limitations / optimisation targets:
 *  - QK redundancy: every thread computes the full D-wide dot product
 *    for each key j.  Replace with a warp reduction over the 64 threads:
 *    each thread contributes q[d]*K[j*D+d], then __shfl_down_sync sums
 *    across the two warps, and the result is broadcast.
 *  - Only 64 threads are in flight; launch multiple blocks to process
 *    independent decode requests in parallel (batched decode).
 *  - For large C, tile K and V into shared memory and use __pipeline
 *    async copies to overlap memory transfers with computation.
 */
__global__ void decode_kernel(const VTYPE *__restrict__ q, /* [D]     */
                              const VTYPE *__restrict__ K, /* [C * D] */
                              const VTYPE *__restrict__ V, /* [C * D] */
                              VTYPE *__restrict__ o,       /* [D]     */
                              int C) {
    /* One block of D threads; each thread d owns one output element. */
    int d = threadIdx.x;
    if (d >= D)
        return;

    float inv_sqrt = 1.f / sqrtf((float)D);
    float out_d = 0.f;
    float m = -FLT_MAX, den = 0.f;

    for (int j = 0; j < C; j++) {
        /*
         * All D threads compute the same dot product — correct because
         * every thread executes the identical sequence of operations and
         * sees the same inputs, so FP results agree exactly.
         * TODO: eliminate redundancy with __shfl_down_sync reduction.
         */
        float score = 0.f;
        for (int dd = 0; dd < D; dd++)
            score += q[dd] * K[QKV(j, dd)];
        score *= inv_sqrt;

        float m_new = (score > m) ? score : m;
        float correction = expf(m - m_new);
        float exp_score = expf(score - m_new);

        den = den * correction + exp_score;
        out_d = out_d * correction + exp_score * V[QKV(j, d)];

        m = m_new;
    }

    o[d] = out_d / den;
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
/* Prefill benchmark                                                   */
/* ------------------------------------------------------------------ */

static void run_prefill(int S,
                        bool skip_standard = false,
                        bool use_standard_kernel = false) {
    long long n = (long long)S * D;
    /* Causal triangle: sum_{i=0}^{S-1}(i+1) * 2D = S*(S+1)*D each for QK
     * and AV
     */
    long long flops = 2LL * S * (S + 1) * D;

    VTYPE *Q = (VTYPE *)malloc(n * sizeof(VTYPE));
    VTYPE *K = (VTYPE *)malloc(n * sizeof(VTYPE));
    VTYPE *V = (VTYPE *)malloc(n * sizeof(VTYPE));
    VTYPE *O_cpu = (VTYPE *)malloc(n * sizeof(VTYPE));
    VTYPE *O_gpu = (VTYPE *)malloc(n * sizeof(VTYPE));

    fill(Q, n, 1);
    fill(K, n, 2);
    fill(V, n, 3);

    printf("PREFILL  S=%d\n", S);

    /* ---- CPU standard_prefill ------------------------------------ */
    if (skip_standard) {
        printf("  %-38s  (O(S^2) cost -- skipped)\n", "CPU standard_prefill");
    } else {
        double t0 = omp_get_wtime();
        standard_prefill(Q, K, V, O_cpu, S);
        double ms = (omp_get_wtime() - t0) * 1e3;
        printf("  %-38s  %8.2f ms\n", "CPU standard_prefill", ms);
    }

    /* ---- CPU flash_prefill (golden reference for GPU verify) ----- */
    {
        double t0 = omp_get_wtime();
        flash_prefill(Q, K, V, O_cpu, S);
        double ms = (omp_get_wtime() - t0) * 1e3;
        printf("  %-38s  %8.2f ms\n", "CPU flash_prefill", ms);
    }

    /* ---- GPU prefill_kernel -------------------------------------- */
    {
        VTYPE *d_Q, *d_K, *d_V, *d_O;
        CUDA_CHECK(cudaMalloc(&d_Q, n * sizeof(VTYPE)));
        CUDA_CHECK(cudaMalloc(&d_K, n * sizeof(VTYPE)));
        CUDA_CHECK(cudaMalloc(&d_V, n * sizeof(VTYPE)));
        CUDA_CHECK(cudaMalloc(&d_O, n * sizeof(VTYPE)));

        CUDA_CHECK(
            cudaMemcpy(d_Q, Q, n * sizeof(VTYPE), cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(d_K, K, n * sizeof(VTYPE), cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(d_V, V, n * sizeof(VTYPE), cudaMemcpyHostToDevice));

        VTYPE *scores;
        if (use_standard_kernel) {
            CUDA_CHECK(cudaMalloc(&scores, S * S * sizeof(VTYPE)));
        }

        /* TODO: tune block size for your kernel */
        // const int BLOCK = D;
        dim3 block(D);
        dim3 grid(S);

        float gpu_ms = 0.f;
        if (use_standard_kernel) {
            /* warmup */
            standard_prefill_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, scores,
                                                     S);
            CUDA_CHECK(cudaDeviceSynchronize());

            /* timed run */
            cudaEvent_t t_start, t_stop;
            CUDA_CHECK(cudaEventCreate(&t_start));
            CUDA_CHECK(cudaEventCreate(&t_stop));
            CUDA_CHECK(cudaEventRecord(t_start));
            standard_prefill_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, scores,
                                                     S);
            CUDA_CHECK(cudaEventRecord(t_stop));
            CUDA_CHECK(cudaEventSynchronize(t_stop));

            CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, t_start, t_stop));
            CUDA_CHECK(cudaEventDestroy(t_start));
            CUDA_CHECK(cudaEventDestroy(t_stop));

            CUDA_CHECK(cudaMemcpy(O_gpu, d_O, n * sizeof(VTYPE),
                                  cudaMemcpyDeviceToHost));

        } else {
            /* warm-up */
            prefill_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, S);
            CUDA_CHECK(cudaDeviceSynchronize());

            /* timed run */
            cudaEvent_t t_start, t_stop;
            CUDA_CHECK(cudaEventCreate(&t_start));
            CUDA_CHECK(cudaEventCreate(&t_stop));
            CUDA_CHECK(cudaEventRecord(t_start));
            prefill_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, S);
            CUDA_CHECK(cudaEventRecord(t_stop));
            CUDA_CHECK(cudaEventSynchronize(t_stop));

            CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, t_start, t_stop));
            CUDA_CHECK(cudaEventDestroy(t_start));
            CUDA_CHECK(cudaEventDestroy(t_stop));

            CUDA_CHECK(cudaMemcpy(O_gpu, d_O, n * sizeof(VTYPE),
                                  cudaMemcpyDeviceToHost));
        }

        bool ok = verify(O_gpu, O_cpu, n);
        double gflops = (double)flops * 1e-9 / (gpu_ms * 1e-3);
        printf("  %-38s  %8.2f ms  %8.2f GFLOPS  %s\n", "GPU prefill_kernel",
               gpu_ms, gflops, ok ? "PASS" : "FAIL");

        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V));
        CUDA_CHECK(cudaFree(d_O));
        if (use_standard_kernel)
            CUDA_CHECK(cudaFree(scores));
    }

    free(Q);
    free(K);
    free(V);
    free(O_cpu);
    free(O_gpu);
    printf("\n");
}

/* ------------------------------------------------------------------ */
/* Decode benchmark                                                    */
/* ------------------------------------------------------------------ */

static void run_decode(int C) {
    /* QK pass: C dot products of length D = 2*C*D FLOPs
       AV pass: same.  Total = 4*C*D FLOPs. */
    long long flops = 4LL * C * D;

    VTYPE *q = (VTYPE *)malloc(D * sizeof(VTYPE));
    VTYPE *K = (VTYPE *)malloc((long long)C * D * sizeof(VTYPE));
    VTYPE *V = (VTYPE *)malloc((long long)C * D * sizeof(VTYPE));
    VTYPE *o_cpu = (VTYPE *)malloc(D * sizeof(VTYPE));
    VTYPE *o_gpu = (VTYPE *)malloc(D * sizeof(VTYPE));

    fill(q, D, 1);
    fill(K, (long long)C * D, 2);
    fill(V, (long long)C * D, 3);

    printf("DECODE   C=%d\n", C);

    /* ---- CPU standard_decode ------------------------------------- */
    {
        double t0 = omp_get_wtime();
        standard_decode(q, K, V, o_cpu, C);
        double ms = (omp_get_wtime() - t0) * 1e3;
        printf("  %-38s  %8.2f ms\n", "CPU standard_decode", ms);
    }

    /* ---- GPU decode_kernel --------------------------------------- */
    {
        VTYPE *d_q, *d_K, *d_V, *d_o;
        CUDA_CHECK(cudaMalloc(&d_q, D * sizeof(VTYPE)));
        CUDA_CHECK(cudaMalloc(&d_K, (long long)C * D * sizeof(VTYPE)));
        CUDA_CHECK(cudaMalloc(&d_V, (long long)C * D * sizeof(VTYPE)));
        CUDA_CHECK(cudaMalloc(&d_o, D * sizeof(VTYPE)));

        CUDA_CHECK(
            cudaMemcpy(d_q, q, D * sizeof(VTYPE), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, K, (long long)C * D * sizeof(VTYPE),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, V, (long long)C * D * sizeof(VTYPE),
                              cudaMemcpyHostToDevice));

        /* D threads per block — one thread per output element */
        dim3 block(D);
        dim3 grid(1);

        /* warm-up */
        decode_kernel<<<grid, block>>>(d_q, d_K, d_V, d_o, C);
        CUDA_CHECK(cudaDeviceSynchronize());

        /* timed run */
        cudaEvent_t t_start, t_stop;
        CUDA_CHECK(cudaEventCreate(&t_start));
        CUDA_CHECK(cudaEventCreate(&t_stop));
        CUDA_CHECK(cudaEventRecord(t_start));
        decode_kernel<<<grid, block>>>(d_q, d_K, d_V, d_o, C);
        CUDA_CHECK(cudaEventRecord(t_stop));
        CUDA_CHECK(cudaEventSynchronize(t_stop));

        float gpu_ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, t_start, t_stop));
        CUDA_CHECK(cudaEventDestroy(t_start));
        CUDA_CHECK(cudaEventDestroy(t_stop));

        CUDA_CHECK(
            cudaMemcpy(o_gpu, d_o, D * sizeof(VTYPE), cudaMemcpyDeviceToHost));

        bool ok = verify(o_gpu, o_cpu, D);
        double gflops = (double)flops * 1e-9 / (gpu_ms * 1e-3);
        printf("  %-38s  %8.2f ms  %8.2f GFLOPS  %s\n", "GPU decode_kernel",
               gpu_ms, gflops, ok ? "PASS" : "FAIL");

        CUDA_CHECK(cudaFree(d_q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V));
        CUDA_CHECK(cudaFree(d_o));
    }

    free(q);
    free(K);
    free(V);
    free(o_cpu);
    free(o_gpu);
    printf("\n");
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(void) {
    int dev;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("=== Attention CUDA  device: %s  D=%d ===\n\n", prop.name, D);

    run_prefill(4096, false, true);
    // run_prefill(65536, /*skip_standard=*/true);
    // run_decode(4096);
    // run_decode(65536);

    return 0;
}
