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
#define Br 16
#define Bc 16

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

    // extern __shared__ VTYPE local_scores[];
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
 *
 * Limitations / optimisation targets:
 *  - No intra-query parallelism: threads do not cooperate on the D-wide
 *    dot products or value accumulations.
 *  - Register pressure: out[D] costs 64 registers per thread, which
 *    limits the number of resident warps.  Consider mapping a thread
 *    block to each query so threads share the work across D.
 *  - Warp-level dot products: use __shfl_down_sync to reduce the D
 *    partial products within a warp rather than looping serially.
 */
__global__ void prefill_kernel(const VTYPE *__restrict__ Q, /* [S * D] */
                               const VTYPE *__restrict__ K, /* [S * D] */
                               const VTYPE *__restrict__ V, /* [S * D] */
                               VTYPE *__restrict__ O,       /* [S * D] */
                               int S) {

    // dim3 block(D, Br): threadIdx.x = d ∈ [0,D-1], threadIdx.y = row ∈ [0,Br-1]
    // Consecutive thread IDs share the same row and have consecutive d → coalesced
    // global/KV reads and output writes.
    int d   = threadIdx.x;  // output dimension [0, D-1]
    int row = threadIdx.y;  // query row within tile [0, Br-1]
    int i   = blockIdx.x * Br + row;

    __shared__ VTYPE Qi[Br * D];
    __shared__ VTYPE Oi[Br * D];
    __shared__ VTYPE Kj[Bc * D];
    __shared__ VTYPE Vj[Bc * D];
    __shared__ VTYPE Sij[Br * Bc];
    // One float per warp per (row, kv_col) for the cross-warp dot-product reduction.
    // Each row spans D/32 = 2 warps; we store one partial per warp per Bc column.
    __shared__ VTYPE dot_warp[Br * (D / 32) * Bc]; // Br * 2 * Bc floats
    __shared__ VTYPE rowmax[Br];
    __shared__ VTYPE rowmaxNew[Br];
    __shared__ VTYPE rsum[Br];
    __shared__ VTYPE rsumNew[Br];
    __shared__ VTYPE corr[Br];

    VTYPE inv_sqrt = 1.f / sqrtf((float)D);

    // load Qi; init Oi — coalesced: same row, consecutive d → Q[i*D .. i*D+D-1]
    Qi[row * D + d] = (i < S) ? Q[QKV(i, d)] : 0.f;
    Oi[row * D + d] = 0.f;

    if (d == 0) {
        rowmax[row] = -INFINITY;
        rsum[row]   = 0.f;
    }
    __syncthreads();

    for (int kvOffset = 0; kvOffset < S; kvOffset += Bc) {
        // Load Kj, Vj — coalesced: threadIdx.y=row ∈ [0,Br-1]==[0,Bc-1],
        // threadIdx.x=d ∈ [0,D-1] → one element per thread, all consecutive.
        int kvGlobal = kvOffset + row;
        Kj[row * D + d] = (kvGlobal < S) ? K[QKV(kvGlobal, d)] : 0.f;
        Vj[row * D + d] = (kvGlobal < S) ? V[QKV(kvGlobal, d)] : 0.f;
        __syncthreads();

        // Compute Sij[row][c] = Qi[row]·Kj[c] for all c ∈ [0,Bc).
        //
        // All D threads per row participate.  Within each row, threads span
        // D/32 = 2 warps.  Strategy:
        //   1. Each thread computes its partial: Qi[row*D+d] * Kj[c*D+d]
        //   2. Warp-level reduction via __shfl_down_sync → lane 0 holds partial sum
        //   3. Lane 0 of each warp writes to dot_warp[row*(D/32)*Bc + warp_in_row*Bc + c]
        //   4. One thread per row assembles the final Sij entry.
        //
        // This keeps all 1024 threads busy vs. the prior 256-active approach.
        {
            int warp_in_row = d / 32;          // 0 or 1 (D=64 → 2 warps per row)
            int lane        = d % 32;
            for (int c = 0; c < Bc; ++c) {
                float partial = Qi[row * D + d] * Kj[c * D + d];
                // Warp-level reduction across 32 lanes
                for (int off = 16; off >= 1; off >>= 1)
                    partial += __shfl_down_sync(0xffffffff, partial, off);
                if (lane == 0)
                    dot_warp[row * (D / 32) * Bc + warp_in_row * Bc + c] = partial;
            }
        }
        __syncthreads();

        // Assemble Sij from warp partials, apply scale and causal mask.
        // Only the first Bc threads in the row (d < Bc) do this.
        if (d < Bc) {
            float acc = 0.f;
            for (int w = 0; w < D / 32; ++w)
                acc += dot_warp[row * (D / 32) * Bc + w * Bc + d];
            int kv_pos = kvOffset + d;
            Sij[row * Bc + d] =
                (i < S && kv_pos <= i) ? acc * inv_sqrt : -INFINITY;
        }
        __syncthreads();

        // Rowmax and correction — one thread per row (d == 0)
        if (d == 0) {
            float mx = rowmax[row];
            for (int k = 0; k < Bc; ++k)
                mx = fmaxf(mx, Sij[row * Bc + k]);
            rowmaxNew[row] = mx;
            corr[row]      = expf(rowmax[row] - mx);
        }
        __syncthreads();

        // Shift + exp Sij
        if (d < Bc)
            Sij[row * Bc + d] = expf(Sij[row * Bc + d] - rowmaxNew[row]);
        __syncthreads();

        // Accumulate row sum — one thread per row
        if (d == 0) {
            VTYPE rowSum = 0.f;
            for (int k = 0; k < Bc; ++k)
                rowSum += Sij[row * Bc + k];
            rsumNew[row] = rsum[row] * corr[row] + rowSum;
        }
        __syncthreads();

        // Value accumulation — all D threads participate, coalesced Vj access.
        // Vj[k*D + d]: fixed k, consecutive d → no bank conflict.
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

    // Normalize and write back — coalesced: same row, consecutive d
    if (i < S)
        O[QKV(i, d)] = Oi[row * D + d] / rsum[row];
}

/*
 * decode_kernel -- flash attention for one query, one thread per output
 * dim.
 *
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

    __shared__ VTYPE Kj[Bc * D];
    __shared__ VTYPE Vj[Bc * D];
    __shared__ VTYPE scores[Bc];
    __shared__ VTYPE correction;
    __shared__ VTYPE maxNew;
    __shared__ VTYPE sumNew;

    VTYPE inv_sqrt = 1.f / sqrtf((float)D);
    float max = -FLT_MAX, sum = 0.f;
    float out_d = 0.f;

    for (int kvOffset = 0; kvOffset < C; kvOffset += Bc) {
        // initialize Kj, Vj
        for (int row = 0; row < Bc; ++row) {
            int kvGlobal = kvOffset + row;
            Kj[QKV(row, d)] = (kvGlobal < C) ? K[QKV(kvGlobal, d)] : 0.f;
            Vj[QKV(row, d)] = (kvGlobal < C) ? V[QKV(kvGlobal, d)] : 0.f;
        }
        __syncthreads();

        // compute score vector
        if (d < Bc) {
            float acc = 0.f;
            for (int k = 0; k < D; ++k)
                acc += q[k] * Kj[QKV(d, k)];
            scores[d] = acc * inv_sqrt;
        }
        __syncthreads();

        // compute max
        if (d == 0) {
            maxNew = max;
            for (int k = 0; k < Bc; ++k)
                maxNew = (maxNew < scores[k]) ? scores[k] : maxNew;
            correction = expf(max - maxNew);
        }
        __syncthreads();

        // shift + exp scores
        if (d < Bc) {
            scores[d] = (kvOffset + d < C) ? expf(scores[d] - maxNew) : 0.f;
        }
        __syncthreads();

        // compute scores sum
        if (d == 0) {
            VTYPE sumRow = 0.f;
            for (int k = 0; k < Bc; ++k)
                sumRow += scores[k];
            sumNew = correction * sum + sumRow;
        }
        __syncthreads();

        // compute output
        float acc = 0.f;
        for (int k = 0; k < Bc; ++k)
            acc += scores[k] * Vj[QKV(k, d)];
        out_d = out_d * correction + acc;

        // forward max, sum
        if (d == 0) {
            max = maxNew;
            sum = sumNew;
        }
        __syncthreads();
    }
    o[d] = out_d / sumNew;
}

/* ------------------------------------------------------------------ */
/* Verify GPU output against CPU reference */
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
/* Prefill benchmark */
/* ------------------------------------------------------------------ */

static void run_prefill(int S,
                        bool skip_standard = false,
                        bool use_standard_kernel = false) {
    long long n = (long long)S * D;
    /* Causal triangle: sum_{i=0}^{S-1}(i+1) * 2D = S*(S+1)*D each for
     * QK and AV
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
        printf("  %-38s  %8.4f ms\n", "CPU standard_prefill", ms);
    }

    /* ---- CPU flash_prefill (golden reference for GPU verify) ----- */
    {
        double t0 = omp_get_wtime();
        flash_prefill(Q, K, V, O_cpu, S);
        double ms = (omp_get_wtime() - t0) * 1e3;
        printf("  %-38s  %8.4f ms\n", "CPU flash_prefill", ms);
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

        float gpu_ms = 0.f;
        if (use_standard_kernel) {
            dim3 block(D);
            dim3 grid(S);
            /* warmup */
            standard_prefill_kernel<<<grid, block, S * sizeof(VTYPE)>>>(
                d_Q, d_K, d_V, d_O, scores, S);
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
            dim3 block(D, Br);   // threadIdx.x=d, threadIdx.y=row → coalesced
            dim3 grid((S + Br - 1) / Br, 1);
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
        printf("  %-38s  %8.4f ms  %8.4f GFLOPS  %s\n", "GPU prefill_kernel",
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
/* Decode benchmark */
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
        printf("  %-38s  %8.4f ms\n", "CPU standard_decode", ms);
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
        printf("  %-38s  %8.4f ms  %8.4f GFLOPS  %s\n", "GPU decode_kernel",
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
/* main */
/* ------------------------------------------------------------------ */

int main(void) {
    int dev;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("=== Attention CUDA  device: %s  D=%d ===\n\n", prop.name, D);

    run_prefill(4096, true);
    run_prefill(65536, /*skip_standard=*/true);
    run_decode(4096);
    run_decode(65536);

    return 0;
}
