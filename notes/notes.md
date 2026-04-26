# Lecture 1.1

## Architecture pre 2010

- economies of scale
    - specialized chips for different machines (smartphone, desktop, etc.)

## trends

- dennard scaling slowdown
    - power wall
    - solution: multi-core

- moore's law slowdown
    - processor cost increases
    - increased physical complexity

- general purpose microarch very hard to improve

- number of cores increasing
    - doesn't solve energy efficiency

## specialization

- hardware: limit capabilities of hardware
- interface: expose hardware through domain-specific interface

## ML/DL hardware

Organized around handful of specific operations

- lots of parallelism at many levels:
    - training (over samples)
    - layers of NN
    - data dimensions
    - operation-level

- memory access:
    - dense: known ahead, contiguous
    - sparse: offline optimizations

- control flow is ~linear

- lots of data reuse

- perfect precision not necessary

## ML for HW

complex policies in CPU:

- branch pred
- mem prefetch
- thread scheduling
- mem access scheduling
- power/freq management

existing approaches are ad hoc

- complicated to design
- brittle
- workload-specific

## Arch quiz

- reg file
    - exposed to s/w, compiler
    - keyed, not indexable (positions in vec regs are indexable)
    - fixed number of regs
- scratchpad memory
    - exposed to s/w, some advanced compilers
    - indexable
    - in GPUs: shared across cores?
- cache
    - hardware managed
    - less common in ML accel

- benefits of caching
    - lower latency
    - lower energy consumption (accessing smaller amount of mem)
    - reduce contention/increase parallelism with cache per core

- SRAM v DRAM
    - DRAM smaller per bit so can fit more on chip
    - DRAM uses more energy
        - SRAM cheaper
    - DRAM off chip
    - DRAM reads destructive

- parallelism:
    - pipelining
    - superscaling
    -

- general purpose overheads:
    - renaming
    - ROB
    - RS
    - control
    - branch pred

- vectorization:
    - helps with branch prediction (like loop unrolling)
    - instruction fetch
    - basically everything...

- caches:
    - L1 should have highest bandwidth, lowest latency
    - if lower caches are shared (e.g. L3), then need increased bandwidth to service several cores

# Lecture 1.2: DL Kernels

## Dot product

- multiplies can be parallelized
- reduction can be O(log n) due to reduction tree

- lacks reuse

## Matrix-vector

```c
type Input[N];
type Weights[M][N];
type Output[M];

for j = 0 -> M { // for each element in output
    for i = 0 -> N { // for each element in input
        Output[j] += Input[i] * Weights[j][i];
    }
}
```

Reuse:

- Input reused M times
- Output accumulator reused N times
- Weights: no reuse

~1 computation per data access

## Training v. Inference

Training requires backward pass.

- similar kernels as forward pass
- must keep forward activations
    - increases memory requirements

### Batching

Advantages of mini-batching:

- (ML) faster than full batch, more stable than SGD
- (Arch) batch requires tons of memory, SGD uses weights once before update, so poor memory access pattern

### Parallelizing training

- over data
- over modules
- need way to reduce updates

## Convolution

CxRxS filters, MxExF output layers (i.e. M filters, ExF 2d output): total computation: CxExSxExFxM

Reuse:

- Filters: ExF
- Fmaps: RxSxM
- Output: RxSxC

Difficulties:

- small kernel -> inefficient, access is uncoalesced.
- not great algos on GPUs (initially).

Can convert to MM using Toeplitz matrix (linearize kernels, lots of redundant data).

## Matrix-Matrix

Batch inputs: QxN. Reuse: M times
Weights: NxM. Reuse: Q times
Output: QxM. Reuse: N times

## Embedding Lookup

Embedding matrices are large -> memory intensive rather than compute intensive

## Elementwise Kernels

- pooling
- normalization

## Specialized Kernels

### Inception

Parallelize convolutions and combine

### Resnet

Identity add needs original input kept in memory

### UNet

Lots of residual connections need to keep data around

### GNNs

Nodes associated with feature embeddings
Message passing: integrate info from neighbors in embedding

- deeper layers incorporate info for further nodes

Sparsity: if graph is not too connected

### RNNs

Mostly MM, but can't parallelize because of temporal dependency

### Transformers

QK^T most computationally intensive operation (scales quadratically with context length)

Inference:

- don't want to recompute attention matrix from scratch during autoregression
    - keep previously computed key matrix
    - keep previously computed values
        - KV cache
    - compute next query, matrix-vector query with cached KVs

Back to matrix-vector -> memory bandwidth becomes dominant again

### Flash Attention

Compute attn matrix one block at a time and use it on the fly to weigh values

# Lecture 2.1

Early GPUs composed of special-purpose units (rasterizer, vertex shader, etc.).

- Low flexibility since pipeline primitives are built into hardware

CUDA released in 2007.

## Differences with CPUs

CPUs exploit _instruction-level parallelism_:

- pipelining
- superscalar
- out of order

Issues:

- high overhead in power and area
- (hence) low computational density

## Sources of parallelism

#### Instruction-Level Parallelism

- Independent instructions within a program.

- Pro: each thread can go faster
- Con: high power/area overhead

#### Thread-Level Parallelism

- Independent threads.
- Hyperthreading: interleave different threads on the same core.

- Pro: can hide memory latency
- Con: programmer/compiler has to find threads to exploit

#### Data-Level Parallelism

- Same operation on several elements (usually of a vector).
- Scale FUs to implement instruction on multiple pieces of data (SIMD)

- Pro: cheap to exploit, just add wider FUs
- Con: harder to program: need to find vector structures on which to perform ops.

## Key properties of GPU programming

- Lots of data parallelism.
    - Many structures are regular.
    - Few dependencies.
- Low locality: data is used for simple operation then unused.

## SIMT

Unified parallelism abstraction.

- Describes both vector and thread-level parallelism

- Group threads into vectors (=warps).
- Warps are what were previously called threads.
- Each thread now processes a single vector element.
- Each warp processes a whole vector.

Programming-level threads are "threads."
Machine-level threads are "warps."

```cpp
gpu void add(int *a, int *b, int *v) {
    c[tid] = a[tib] + b[tid];
}
```

Functional perspective:

- Each thread is a real thread (owns PC and register file)

Performance perspective:

- All threads in a warp (ought to) have same PC

What about control flow?

```cpp
if (tid%2) {
    c[tid] = a[tid] + b[tid]
} else {
    c[tid] = a[tid] - b[tid]
}
```

Execute both branches, masking threads not selected by branch.

- Introduces hardware overhead; need divergence stack keeping track of which branches need to be executed and where/how they should rejoin.

## Memory

Traditional caches require one lookup per cycle (per port).

- Prohibitive for massively multithreaded GPU.
    - Either too slow or too expensive (many ports).

Instead, use address coalescing.

- Exploits fact that threads in same warp usually access same cache blocks/lines.

### Downside of caches

- Not energy efficient
    - Complex coalescing logic
    - Tag-checking is overhead
    - Cache replacement can be complicated

Use software to manage data reuse?

## Scratchpads

Small, private, software-managed cache.

- Separate address space shared between some threads.
- Fully software-managed.
- High bandwidth.
- Low energy cost.
- No need for coalescing, better for irregular access.

Scratchpad organized into indexed banks.

- Each thread can access each bank.
- If accesses are independent, all accesses can be done in a single cycle.
- If conflicts, need to serialize access.
    - Strided access pattern can cause conflicts.

### Sharing scratchpads

- All cores share a single scratchpad.
    - Easy to program
    - High latency?
- Each core has own scratchpad
    - No sharing
- Hierarchy: organize warps into blocks,which can share a scratchpad. Blocks are organized into a grid.

Hierarchy:

- Global memory (DRAM, very high bandwidth)
- L2$
- SM
    - Registers
    - L1$
    - Cores
        - Grid
            - Block
                - Scratchpad
                - Warp
                    - Thread

GPU<->CPU data transfer often a bottleneck.

# Lecture 2.2

## Programming GPUs

Single Program Multiple Data

### Low Level SPMD

- CUDA (NVIDIA)
- HIP (AMD)

### High-Level SPMD

- Triton (OpenAI): Python DSL, tile-level instead of thread-level
- Kokkos (Sandia): header files on top of CUDA/HIP
- SYCL (Khronos)

### Libraries

- cuBLAS
- cuDNN

## CUDA Programming

Extensions to C/C++ for heterogeneous computing

- Manage memory
- Manage kernels

Host: CPU + memory
Device: GPU + memory

```cpp
// __global__ indicates code runs on device, called from host
// non-__global__ code compiled by gcc
// __global__ code compiled by nvcc
__global__ void add(int *a , int *b, int *c) {

}

int main(void){
    int a, b, c; // host copies
    int *d_a, *d_b, *d_c; // device copies
    int size = sizeof(int);

    // allocate arrays to GPU memory
    cudaMalloc((void **) &d_a, size);
    cudaMalloc((void **) &d_b, size);
    cudaMalloc((void **) &d_c, size);

    int nBlocks
    add<<<1, 1>>>();
    printf("Hello World\n");
    return 0;
}
```

Memory management:

- `cudaMalloc()`, `cudaFree()`, `cudaMemcpy()`
    - Note `cudaMemcpy()` can copy from host to device and vice versa.
- Want _warp-level contiguity_ in memory accesses: contiguous threads access contiguous memory

### Parallelization

Blocks:

`add<<<N, 1>>>`

- want at least one block per core

Threads:

Instead of gating data access (when size is not a multiple of 32), pad array.

- Less risk of error
- No overhead for guard

### Cooperating threads

Consider 1d stencil on 1d array.

Radius: # of elements above/below required for computation of stencil at i.

With radius n, each element (except edges) used n times.

Suppose each thread processes one element of output.

- Want threads in a block to share accesses to input elements.
- Can share access using scratchpad (aka shared memory)
- Declare variables using `__shared__`.

## Cache Coherence

Not supposed to communicate between blocks.

- L1$ may be incoherent

Coherence can be restored:

- `threadfence()` will flush writes to L2 (but will not update other L1s)
- `volatile` prevents caching in L1

## Unified Memory

Software-managed memory access:

- `cuda_malloc`: allocate device memory
- `cuda_memcpy`: transfer from host to device
    - This makes pointers hard!

Unified Memory:

- `cudaMallocManaged`: allocates transparently managed memory

GraceHopper:

- cache coherence between CPU and GPU
    - single page table

## Why do GPUs work?

- eliminate overhead from OOO
- vector execution (SIMT)
- fine-grained software-level control of local access (scratchpad)

What else?

Specialized hw:

- dot product
- Tensor cores
    - operate at warp level
        - break SIMT model
    - composed of subcores
        - register file (loaded as array)

Hopper SM:

- support for low-precision
- New tensor cores
- dynamic programming instructions

Sparsity

- wasteful to load and compute over 0s
- structured vs unstructured sparsity

## Global shared memory

Introduce thread block clusters

- Sits between grid and block in mem hierarchy
- Equipped with scratchpad
- Guarantees concurrency for blocks within cluster
- A block cluster can be scheduled across SMs

## Tensor Memory Accelerator

dedicated memory instruction unit

## Why are GPUs not solved yet?

Overheads:

- Memory access:
    - dynamic coalescing energy overheads
    - cache thrashing
- Control flow
    - need to track divergence
- Operand communication
- Scheduling
    - Warps/threads need to be dynamically scheduled

# Lecture 3.1: Data Reuse

Workload scenario:

- CNN: conv, pooling, FC

### Baking DNN into hardware

- super fast (fully pipelined --> one inf per cycle)
- not scalable
    - literally too big
- not modular

### DianNao

- optimized for 16x16 mm.

Pooling computations:

- Nx _ Ny _ Ni _ Kx _ Ky / (Sx \* Sy)

Conv:

- Ox _ Oy _ On _ Kx _ Ky / (Sx \* Sy)

## Reuse

More reuse --> less memory bandwidth required

Best-case reuse: total data access / total computations

- total compute: prod dims.
- total data access: volume of access (?)
- achievable?
    - total compute: yes if no dependencies between data.
    - total data access: not always: not all data can be reused immediately.
        - local memory constraints?

Achievable reuse

- hierarchically decompose problem
    - natural breaking points: loop nesting

## DianNao

What's the data layout?

- In what order do you nest the loops?

"that is the payload of what I'm saying"

DianNao conv optimizes dataflow for reuse over inputs, not weights.

- Want to instead ensure some reuse across both

## Scratchpad vs cache

Scratchpad memory can be more precisely controlled.

- Cache is LRU or similar.

## Summary

- demonstrates performance gains for simple architectural modifications
- avoids overheads associated with CPUs:
    - no control flow
    - lots of parallelism
    - simpe ISA
        - limits what the accel can do
- massive benefits from memory
    - increased reuse opportunity
- reduction rather than SIMD:
    - less output storage required
- training vs inference
    - training has higher mem reqs: need to keep activations around for backwards pass

# Lecture 3.2 - Eyeriss

Importance of Eyeriss:

- Defines dataflow design space problem

Main problem:

- CNNs require lots of mem reads
- Lots of reuse to exploit
    - Input pixels
    - Weights (filters)
    - Output (partial sums)
- How do you exploit all these forms of reuse?

Reuse as resource:

- Temporal: Using the same data at different (sequential?) time steps
- Spatial: Using the same data at different locations in
    - Can distribute a given amount of reuse temporally or spatially

Memory hierarchy:

- want reuse at higher levels to prevent accesses at lower levels
- relative reuse ratio: for each access at level n, how many times is level n+1 accessed?

Dataflow:

- parallelization strategy (+ tiling) + mapping to PEs

Opportunities for reuse in conv layers:

- conv reuse: activations + filter weights
- fmap: reuse activations
- filter: filter weights (if batched inputs)

X-Stationary Dataflow

- What do we keep in the X?
    - keep output in X -> output is X-stationary
    - keep weights in X -> weight is X-stationary
    - ...

# Lecture 4.2 - Sparsity

How to reduce off-chip memory bandwidth?

- On-chip buffers (cache, scratchpad)
    - need to strike balance between size, cost, latency

- Change DNN architecture
- Quantize weights
- Sparsity
    - Some weights less important than others!

## Non-uniform importance

- some weights are small, or otherwise don't affect result
- don't need a very diverse range of weights
- get rid of zero-weights?

## Forms of sparsity

### Static sparsity

Some weights may be removed.

### Dynamic sparsity

Negative activations in ReLU are 0.

When both forms of sparsity present (sparse weights, sparse activations):

- space saving
- computational saving

Requires sparse MM, harder to optimize

Requires re-encoding

## Sparsification

- Post-training pruning
    - train dense model, prune, then retrain
    - expensive due to retraining, so used on small models

- Prune during training
    - weights zeroed progressively during training
    - more adaptive but still expensive

- One shot pruning
    - the above are too expensive for modern LLMs
    - score and prune pretrained model in single-pass (no retraining)
        - magnitude pruning: remove weights that are close to 0
        - wanda: do a forward pass on a small number of samples, score on average magnitude of output

Weight pruning:

- positive: less storage , less comp
- Negative: mem overhead for sparsity, additional comp to access, poor for SIMD, tiling (since noncontiguous)

## Kinds of sparsity

- unstructured
- block-sparse (coarse grained)
- window-based sparsity (KxN)
- patterned...

## Representation

- coordinate: store non-sparse values in vector, x and y coordinates in separate vectors
    - more memory efficient to flatten coordinates to 1D
- run length: non-sparse values together with run length of prior 0s
- bitmap: store non-sparse together with bitmap of whether value non-sparse
