# Setup

We'll consider kernels with signatures that are variations on the following:

```cpp
__global__ void standard_prefill_kernel(
    const VTYPE *__restrict__ Q, /* [S * D] */
    const VTYPE *__restrict__ K, /* [S * D] */
    const VTYPE *__restrict__ V, /* [S * D] */
    VTYPE *__restrict__ O,       /* [S * D] */
    VTYPE *__restrict__ scores,  /* [S * S] workspace — caller allocates */
    int S);
```
Here, `S` is the sequence length, `D` is the model dimension, `Q`, `K`, `V`, and `O` are the key, query, value, and output matrices, and `scores` is a buffer  that will hold the softmax scores.

We assume that the model dimension $D$ is available as a global variable.

We'll also use a macro to index into flat array as if it were two-dimensional:

```
#define QKV(row, col) ((row) * D + (col))
```

# Naive implementation

In this implementation, we let each thread block handle a query and each thread handle an output element: thread $(i, d)$ computes $O_{id}$.

The first several steps compute the attention scores for query $i$.
Since the attention scores are shared between all threads in a block, we let a single thread compute them by gating the computation behind `if (d == 0) ...`.
To ensure that the scores have been computed before they are used, we use `__syncthreads` before averaging over values.

```cpp
__global__ void standard_prefill_kernel(
    const VTYPE *__restrict__ Q, /* [S * D] */
    const VTYPE *__restrict__ K, /* [S * D] */
    const VTYPE *__restrict__ V, /* [S * D] */
    VTYPE *__restrict__ O,       /* [S * D] */
    VTYPE *__restrict__ scores,  /* [S * S] workspace — caller allocates */
    int S) {

    // S blocks, each containing D threads
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
```

# Better 

The naive implementation is inefficient in a number of ways.
We can do better 
