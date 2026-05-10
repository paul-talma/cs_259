#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <iterator>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_e));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

void initialize(float *t, float val, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        t[i] = val;
    }
}
void displayResult(float *tensor, int width, int bound) {
    for (size_t i = 0; i < bound; ++i) {
        for (size_t j = 0; j < bound; ++j) {
            std::cout << tensor[i * width + j];
            if (j < 4)
                std::cout << ", ";
        }
        std::cout << "\n";
    }
}

__global__ void conv2d(float *padded_input,
                       float *filter,
                       float *output,
                       int paddedInputWidth,
                       int paddedInputHeight,
                       int filterWidth,
                       int filterHeight,
                       int outputWidth,
                       int outputHeight,
                       int inChannels,
                       int outChannels,
                       int batchSize,
                       int stride) {
    // get current thread location
    size_t outputCol = blockDim.x * blockIdx.x + threadIdx.x;
    size_t outputRow = blockDim.y * blockIdx.y + threadIdx.y;

    if (outputCol >= outputWidth || outputRow >= outputHeight)
        return;
    // batch
    for (size_t inputNumber = 0; inputNumber < batchSize; ++inputNumber) {
        // out channels
        for (size_t outChannel = 0; outChannel < outChannels; ++outChannel) {
            // in channels
            for (size_t inChannel = 0; inChannel < inChannels; ++inChannel) {
                float acc = 0;
                // conv
                for (size_t filterRow = 0; filterRow < filterHeight;
                     ++filterRow) {
                    for (size_t filterCol = 0; filterCol < filterWidth;
                         ++filterCol) {
                        size_t flatInputIdx =
                            inputNumber * (inChannels * paddedInputHeight *
                                           paddedInputWidth) + // each input is
                                                               // CxHPADxWPAD

                            inChannel * (paddedInputHeight * // each channel is
                                         paddedInputWidth) + // HPADxWPAD

                            (outputRow * stride + filterRow) *
                                paddedInputWidth + // each row is WPAD

                            (outputCol * stride + filterCol);

                        size_t filterId =
                            outChannel *
                                (inChannels * filterHeight * filterWidth) +
                            inChannel * (filterHeight * filterWidth) +
                            filterRow * filterWidth + filterCol;

                        acc += padded_input[flatInputIdx] * filter[filterId];
                    }
                }
                size_t flatOutputIdx =
                    inputNumber * // each output is  oCxHxW
                        (outChannels * outputHeight * outputWidth) +

                    outChannel * // each channel is HxW
                        (outputHeight * outputWidth) +

                    outputRow * outputWidth + // each row is W

                    outputCol;
                output[flatOutputIdx] += acc;
            }
        }
    }
}

// CPU reference — same memory layout as the GPU kernel:
//   input  : [batchSize][inChannels][paddedHeight][paddedWidth]
//   filter : [outChannels][inChannels][filterHeight][filterWidth]
//   output : [batchSize][outChannels][outputHeight][outputWidth]
//
// NOTE: when stride is wired into the GPU kernel, apply the same
//       change here (outputRow*stride, outputCol*stride).
void conv2d_ref(const float *input,
                const float *filter,
                float *output,
                int paddedInputWidth,
                int paddedInputHeight,
                int filterWidth,
                int filterHeight,
                int outputWidth,
                int outputHeight,
                int inChannels,
                int outChannels,
                int batchSize,
                int stride) {
    for (int n = 0; n < batchSize; ++n) {
        for (int oc = 0; oc < outChannels; ++oc) {
            for (int row = 0; row < outputHeight; ++row) {
                for (int col = 0; col < outputWidth; ++col) {
                    float acc = 0.0f;
                    for (int ic = 0; ic < inChannels; ++ic) {
                        for (int fy = 0; fy < filterHeight; ++fy) {
                            for (int fx = 0; fx < filterWidth; ++fx) {
                                int inIdx =
                                    n * (inChannels * paddedInputHeight *
                                         paddedInputWidth) +
                                    ic *
                                        (paddedInputHeight * paddedInputWidth) +
                                    (row * stride + fy) * paddedInputWidth +
                                    (col * stride + fx);
                                int filtIdx =
                                    oc * (inChannels * filterHeight *
                                          filterWidth) +
                                    ic * (filterHeight * filterWidth) +
                                    fy * filterWidth + fx;
                                acc += input[inIdx] * filter[filtIdx];
                            }
                        }
                    }
                    int outIdx =
                        n * (outChannels * outputHeight * outputWidth) +
                        oc * (outputHeight * outputWidth) + row * outputWidth +
                        col;
                    output[outIdx] = acc;
                }
            }
        }
    }
}

bool verify(const float *gpu, const float *ref, size_t n, float tol = 1e-3f) {
    size_t failures = 0;
    float maxErr = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float err = fabsf(gpu[i] - ref[i]);
        if (err > maxErr)
            maxErr = err;
        if (err > tol) {
            if (failures < 5) // print first few mismatches
                fprintf(stderr,
                        "  mismatch at [%zu]: gpu=%.6f  ref=%.6f  err=%.2e\n",
                        i, gpu[i], ref[i]);
            ++failures;
        }
    }
    if (failures == 0) {
        printf("PASS  (%zu elements, max err = %.2e)\n", n, maxErr);
        return true;
    } else {
        fprintf(stderr,
                "FAIL  %zu/%zu elements exceed tol=%.2e  (max err=%.2e)\n",
                failures, n, tol, maxErr);
        return false;
    }
}

// Run one test case. Returns true if GPU output matches CPU reference.
bool runTest(const char *label,
             int inputWidth,
             int inputHeight,
             int filterWidth,
             int padding,
             int stride,
             int inChannels,
             int outChannels,
             int batchSize) {
    printf("%-40s ... ", label);
    fflush(stdout);

    int outputWidth =
        (inputWidth + 2 * padding - (filterWidth - 1) - 1) / stride + 1;
    int outputHeight =
        (inputHeight + 2 * padding - (filterWidth - 1) - 1) / stride + 1;
    int padW = inputWidth + 2 * padding;
    int padH = inputHeight + 2 * padding;

    size_t nIn = (size_t)batchSize * inChannels * padH * padW;
    size_t nF = (size_t)outChannels * inChannels * filterWidth * filterWidth;
    size_t nOut = (size_t)batchSize * outChannels * outputHeight * outputWidth;

    float *h_in = (float *)malloc(nIn * sizeof(float));
    float *h_f = (float *)malloc(nF * sizeof(float));
    float *h_out = (float *)malloc(nOut * sizeof(float));
    float *h_ref = (float *)malloc(nOut * sizeof(float));

    // Non-uniform values so that index bugs produce wrong results
    for (size_t i = 0; i < nIn; ++i)
        h_in[i] = (float)(i % 17) * 0.1f - 0.8f;
    for (size_t i = 0; i < nF; ++i)
        h_f[i] = (float)(i % 7) * 0.2f - 0.6f;

    // CPU reference
    conv2d_ref(h_in, h_f, h_ref, padW, padH, filterWidth, filterWidth,
               outputWidth, outputHeight, inChannels, outChannels, batchSize,
               stride);

    // GPU
    float *d_in, *d_f, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, nIn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_f, nF * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, nOut * sizeof(float)));
    CUDA_CHECK(
        cudaMemcpy(d_in, h_in, nIn * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_f, h_f, nF * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, nOut * sizeof(float)));

    dim3 threads(16, 16);
    dim3 blocks((outputWidth + threads.x - 1) / threads.x,
                (outputHeight + threads.y - 1) / threads.y);
    conv2d<<<blocks, threads>>>(d_in, d_f, d_out, padW, padH, filterWidth,
                                filterWidth, outputWidth, outputHeight,
                                inChannels, outChannels, batchSize, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(
        cudaMemcpy(h_out, d_out, nOut * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = verify(h_out, h_ref, nOut);

    cudaFree(d_in);
    cudaFree(d_f);
    cudaFree(d_out);
    free(h_in);
    free(h_f);
    free(h_out);
    free(h_ref);
    return ok;
}

int main(void) {
    bool allPassed = true;

    //                label                            W    H   FW  pad  stride
    //                Ci  Co   B
    allPassed &= runTest("1x1x1  3x3 filter  stride=1", 8, 8, 3, 1, 1, 1, 1, 1);
    allPassed &=
        runTest("1x1x1  5x5 filter  stride=1", 16, 16, 5, 2, 1, 1, 1, 1);
    allPassed &= runTest("multi-channel Ci=3 Co=4", 16, 16, 3, 1, 1, 3, 4, 1);
    allPassed &= runTest("batch=4  Ci=2  Co=3", 16, 16, 3, 1, 1, 2, 3, 4);
    allPassed &= runTest("non-square input 32x16", 32, 16, 3, 1, 1, 2, 2, 2);

    // stride=2 tests
    //                label                                W    H   FW pad
    //                stride Ci Co  B
    allPassed &=
        runTest("stride=2  3x3  no-pad  Ci=1 Co=1", 32, 32, 3, 0, 2, 1, 1, 1);
    allPassed &=
        runTest("stride=2  3x3  pad=1   Ci=3 Co=4", 32, 32, 3, 1, 2, 3, 4, 2);
    allPassed &=
        runTest("stride=2  5x5  pad=2   Ci=2 Co=4", 32, 32, 5, 2, 2, 2, 4, 2);
    allPassed &=
        runTest("stride=2  non-square 64x32", 64, 32, 3, 1, 2, 2, 2, 1);
    allPassed &=
        runTest("stride=2  batch=4  Ci=3 Co=8", 16, 16, 3, 1, 2, 3, 8, 4);

    // stride=3 tests
    allPassed &=
        runTest("stride=3  3x3  no-pad  Ci=1 Co=1", 48, 48, 3, 0, 3, 1, 1, 1);
    allPassed &=
        runTest("stride=3  3x3  pad=1   Ci=2 Co=4", 48, 48, 3, 1, 3, 2, 4, 2);
    allPassed &=
        runTest("stride=3  5x5  pad=2   Ci=1 Co=2", 30, 30, 5, 2, 3, 1, 2, 1);

    // stride=4 test
    allPassed &=
        runTest("stride=4  3x3  no-pad  Ci=1 Co=1", 64, 64, 3, 0, 4, 1, 1, 1);

    return allPassed ? 0 : 1;
}
