#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define D_SIZE 64
#define C_SIZE 65536
#define ROW_OFFSETS 4
#define BLOCK_THREADS_X 64
#define ROWS_PER_BLOCK (BLOCK_THREADS_X * ROW_OFFSETS)
#define CPU_BENCH_ITERS 20
#define GPU_BENCH_ITERS 1000

static void check_cuda(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error during " << what << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static float dot_cpu(const float *a, const float *b) {
    float sum = 0.0f;
    for (int i = 0; i < D_SIZE; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

static void fill(float *m, long long n, int seed) {
    for (long long i = 0; i < n; i++) {
        m[i] = 0.1f * sinf(static_cast<float>(i * 3 + seed * 7));
    }
}

static void standard_decode_cpu(
    const float *q, const float *K, const float *V, float *o, int C) {
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(D_SIZE));
    std::vector<float> scores(C);

    for (int j = 0; j < C; j++) {
        scores[j] = dot_cpu(q, K + j * D_SIZE) * inv_sqrt;
    }

    float mx = scores[0];
    for (int j = 1; j < C; j++) {
        mx = std::max(mx, scores[j]);
    }

    float sum = 0.0f;
    for (int j = 0; j < C; j++) {
        scores[j] = std::exp(scores[j] - mx);
        sum += scores[j];
    }

    for (int d = 0; d < D_SIZE; d++) {
        float acc = 0.0f;
        for (int j = 0; j < C; j++) {
            acc += (scores[j] / sum) * V[j * D_SIZE + d];
        }
        o[d] = acc;
    }
}

__global__ void flash_decode(const float *q,
                             const float *K,
                             const float *V,
                             float *o,
                             float *tmp_out,
                             float *tmp_stats,
                             int C,
                             int D) {
    __shared__ float query[D_SIZE];
    __shared__ float s_bridge[ROW_OFFSETS][2];

    int d = threadIdx.x;
    int row_off = threadIdx.y;
    if (row_off == 0) {
        query[d] = q[d];
    }
    __syncthreads();

    int bid = blockIdx.x;
    int rows_per_block = blockDim.x * blockDim.y;

    float inv_sqrt = 1.0f / sqrtf(static_cast<float>(D));
    float local_max = -1e20f;
    float local_sum = 0.0f;
    float local_acc = 0.0f;

    for (int i = 0; i < blockDim.x; i++) {
        int row_idx = bid * rows_per_block + (i * blockDim.y) + row_off;
        bool valid_row = row_idx < C;
        float s = valid_row ? query[d] * K[row_idx * D + d] : 0.0f;
        float score = s;

        for (int offset = 16; offset > 0; offset /= 2) {
            score += __shfl_down_sync(0xFFFFFFFF, score, offset);
        }

        int lane = d % 32;
        int warp_id = d / 32;
        if (lane == 0) {
            s_bridge[row_off][warp_id] = score;
        }
        __syncthreads();

        if (row_idx < C) {
            s = (s_bridge[row_off][0] + s_bridge[row_off][1]) * inv_sqrt;

            float old_max = local_max;
            local_max = fmaxf(old_max, s);
            float exp_score = expf(s - local_max);
            float rescale = expf(old_max - local_max);

            local_sum = local_sum * rescale + exp_score;
            local_acc = local_acc * rescale + exp_score * V[row_idx * D + d];
        }
        __syncthreads();
    }

    __shared__ float final_maxs[ROW_OFFSETS];
    __shared__ float final_sums[ROW_OFFSETS];
    __shared__ float final_accs[ROW_OFFSETS][D_SIZE];

    if (d == 0) {
        final_maxs[row_off] = local_max;
        final_sums[row_off] = local_sum;
    }
    final_accs[row_off][d] = local_acc;
    __syncthreads();

    if (row_off == 0) {
        float b_max = -1e20f;
        for (int j = 0; j < ROW_OFFSETS; j++) {
            b_max = fmaxf(b_max, final_maxs[j]);
        }

        float b_sum = 0.0f;
        float b_acc = 0.0f;
        for (int j = 0; j < ROW_OFFSETS; j++) {
            float scale = expf(final_maxs[j] - b_max);
            b_sum += final_sums[j] * scale;
            b_acc += final_accs[j][d] * scale;
        }

        tmp_out[bid * D + d] = b_acc;
        if (d == 0) {
            tmp_stats[bid * 2] = b_max;
            tmp_stats[bid * 2 + 1] = b_sum;
        }
    }

    (void)o;
}

__global__ void finalize_flash_decode(const float *tmp_out,
                                      const float *tmp_stats,
                                      float *o,
                                      int num_blocks,
                                      int D) {
    int d = threadIdx.x;

    float global_max = -1e20f;
    for (int b = 0; b < num_blocks; b++) {
        global_max = fmaxf(global_max, tmp_stats[b * 2]);
    }

    float global_sum = 0.0f;
    float acc = 0.0f;
    for (int b = 0; b < num_blocks; b++) {
        float scale = expf(tmp_stats[b * 2] - global_max);
        global_sum += tmp_stats[b * 2 + 1] * scale;
        acc += tmp_out[b * D + d] * scale;
    }

    o[d] = acc / global_sum;
}

static void run_flash_decode_comparison(int C) {
    constexpr int D = D_SIZE;
    const int num_blocks = (C + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const double decode_flops = 4.0 * C * D;

    std::vector<float> h_q(D);
    std::vector<float> h_K(static_cast<size_t>(C) * D);
    std::vector<float> h_V(static_cast<size_t>(C) * D);
    std::vector<float> h_cpu_o(D, 0.0f);
    std::vector<float> h_gpu_o(D, 0.0f);

    fill(h_q.data(), D, 1);
    fill(h_K.data(), static_cast<long long>(C) * D, 2);
    fill(h_V.data(), static_cast<long long>(C) * D, 3);

    standard_decode_cpu(h_q.data(), h_K.data(), h_V.data(), h_cpu_o.data(), C);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < CPU_BENCH_ITERS; iter++) {
        standard_decode_cpu(h_q.data(), h_K.data(), h_V.data(), h_cpu_o.data(),
                            C);
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_total_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    double cpu_avg_ms = cpu_total_ms / CPU_BENCH_ITERS;
    double cpu_gflops = decode_flops / (cpu_avg_ms / 1000.0) / 1.0e9;

    float *d_q = nullptr;
    float *d_K = nullptr;
    float *d_V = nullptr;
    float *d_o = nullptr;
    float *d_tmp_out = nullptr;
    float *d_tmp_stats = nullptr;

    check_cuda(cudaMalloc(&d_q, D * sizeof(float)), "cudaMalloc d_q");
    check_cuda(cudaMalloc(&d_K, static_cast<size_t>(C) * D * sizeof(float)),
               "cudaMalloc d_K");
    check_cuda(cudaMalloc(&d_V, static_cast<size_t>(C) * D * sizeof(float)),
               "cudaMalloc d_V");
    check_cuda(cudaMalloc(&d_o, D * sizeof(float)), "cudaMalloc d_o");
    check_cuda(cudaMalloc(&d_tmp_out,
                          static_cast<size_t>(num_blocks) * D * sizeof(float)),
               "cudaMalloc d_tmp_out");
    check_cuda(cudaMalloc(&d_tmp_stats,
                          static_cast<size_t>(num_blocks) * 2 * sizeof(float)),
               "cudaMalloc d_tmp_stats");

    check_cuda(
        cudaMemcpy(d_q, h_q.data(), D * sizeof(float), cudaMemcpyHostToDevice),
        "copy q");
    check_cuda(cudaMemcpy(d_K, h_K.data(),
                          static_cast<size_t>(C) * D * sizeof(float),
                          cudaMemcpyHostToDevice),
               "copy K");
    check_cuda(cudaMemcpy(d_V, h_V.data(),
                          static_cast<size_t>(C) * D * sizeof(float),
                          cudaMemcpyHostToDevice),
               "copy V");

    dim3 block(BLOCK_THREADS_X, ROW_OFFSETS);
    flash_decode<<<num_blocks, block>>>(d_q, d_K, d_V, d_o, d_tmp_out,
                                        d_tmp_stats, C, D);
    check_cuda(cudaGetLastError(), "flash_decode launch");
    finalize_flash_decode<<<1, D>>>(d_tmp_out, d_tmp_stats, d_o, num_blocks, D);
    check_cuda(cudaGetLastError(), "finalize_flash_decode launch");
    check_cuda(cudaDeviceSynchronize(), "kernel execution");

    cudaEvent_t gpu_start;
    cudaEvent_t gpu_stop;
    check_cuda(cudaEventCreate(&gpu_start), "create start event");
    check_cuda(cudaEventCreate(&gpu_stop), "create stop event");

    check_cuda(cudaEventRecord(gpu_start), "record start event");
    for (int iter = 0; iter < GPU_BENCH_ITERS; iter++) {
        flash_decode<<<num_blocks, block>>>(d_q, d_K, d_V, d_o, d_tmp_out,
                                            d_tmp_stats, C, D);
        finalize_flash_decode<<<1, D>>>(d_tmp_out, d_tmp_stats, d_o, num_blocks,
                                        D);
    }
    check_cuda(cudaEventRecord(gpu_stop), "record stop event");
    check_cuda(cudaEventSynchronize(gpu_stop), "synchronize stop event");
    check_cuda(cudaGetLastError(), "benchmark kernel launches");

    float gpu_total_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&gpu_total_ms, gpu_start, gpu_stop),
               "measure GPU elapsed time");
    double gpu_avg_ms = static_cast<double>(gpu_total_ms) / GPU_BENCH_ITERS;
    double gpu_gflops = decode_flops / (gpu_avg_ms / 1000.0) / 1.0e9;

    check_cuda(cudaMemcpy(h_gpu_o.data(), d_o, D * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "copy output");

    float max_abs_err = 0.0f;
    float mean_abs_err = 0.0f;
    int max_idx = 0;
    for (int i = 0; i < D; i++) {
        float err = std::fabs(h_cpu_o[i] - h_gpu_o[i]);
        mean_abs_err += err;
        if (err > max_abs_err) {
            max_abs_err = err;
            max_idx = i;
        }
    }
    mean_abs_err /= D;

    constexpr float tol = 1e-5f;
    std::cout << "=== Flash decode comparison ===\n";
    std::cout << "D=" << D << " C=" << C << " blocks=" << num_blocks << "\n";
    std::cout << "FLOP estimate: 4*C*D = " << decode_flops
              << " FLOPs per decode\n";
    std::cout << "CPU avg time: " << cpu_avg_ms << " ms over "
              << CPU_BENCH_ITERS << " iterations, " << cpu_gflops
              << " GFLOP/s\n";
    std::cout << "GPU avg kernel time: " << gpu_avg_ms << " ms over "
              << GPU_BENCH_ITERS << " iterations, " << gpu_gflops
              << " GFLOP/s\n";
    std::cout << "max_abs_err=" << max_abs_err << " at o[" << max_idx << "]\n";
    std::cout << "mean_abs_err=" << mean_abs_err << "\n";
    std::cout << "First 8 outputs:\n";
    for (int i = 0; i < 8; i++) {
        std::cout << "o[" << i << "] CPU=" << h_cpu_o[i]
                  << " GPU=" << h_gpu_o[i]
                  << " abs_err=" << std::fabs(h_cpu_o[i] - h_gpu_o[i]) << "\n";
    }
    std::cout << (max_abs_err <= tol ? "PASS" : "FAIL")
              << ": compared against CPU standard_decode, tolerance " << tol
              << ".\n";

    cudaFree(d_q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_o);
    cudaFree(d_tmp_out);
    cudaFree(d_tmp_stats);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
}

int main() {
    run_flash_decode_comparison(C_SIZE);
    return 0;
}
