// oxmera CUDA kernels: strided elementwise, axis reductions, block-level
// full reduction, tiled matmul, and gather/scatter along one dimension — a
// one-to-one port of kernels.metal
// (same TensorMeta ABI, same opcodes) so the two GPU backends share one
// host-side contract. f32 throughout; strided access is described by
// TensorMeta (dims/strides/offset up to rank 8), with stride 0 encoding a
// broadcast dimension.
//
// Compiled to PTX for the `compute_75` virtual architecture (Turing and
// newer, JIT-forward-compatible) and embedded in the crate:
//
//     nvcc -arch=compute_75 -O3 -ptx kernels.cu -o kernels.ptx
//
// The `.cu` source is also embedded, so a driver that rejects the checked-in
// PTX ISA version can rebuild it at runtime through NVRTC.

#define MAX_RANK 8

struct TensorMeta {
    unsigned int rank;
    unsigned int offset;
    unsigned int dims[MAX_RANK];
    int strides[MAX_RANK];
};

// Storage offset of logical element `gid` (row-major over the dims).
__device__ __forceinline__ unsigned int strided_offset(unsigned int gid, const TensorMeta& m) {
    unsigned int rem = gid;
    int off = (int)m.offset;
    for (unsigned int i = 0; i < m.rank; i++) {
        unsigned int d = m.rank - 1 - i;
        unsigned int dim = m.dims[d];
        unsigned int coord = rem % dim;
        rem /= dim;
        off += (int)coord * m.strides[d];
    }
    return (unsigned int)off;
}

// ---- elementwise -----------------------------------------------------------

__device__ __forceinline__ float unary_eval(unsigned int op, float x) {
    switch (op) {
        case 0: return x;                       // copy / contiguous
        case 1: return -x;                      // neg
        case 2: return expf(x);                 // exp
        case 3: return logf(x);                 // ln
        case 4: return fabsf(x);                // abs
        case 5: return sqrtf(x);                // sqrt
        case 6: return sinf(x);                 // sin
        case 7: return cosf(x);                 // cos
        case 8: return tanhf(x);                // tanh
        case 9: return fmaxf(x, 0.0f);          // relu
        case 10: {                              // gelu (tanh approximation)
            float u = 0.7978845608f * (x + 0.044715f * x * x * x);
            return 0.5f * x * (1.0f + tanhf(u));
        }
        case 11: return 1.0f / (1.0f + expf(-x)); // sigmoid
        default: return nanf("");
    }
}

__device__ __forceinline__ float binary_eval(unsigned int op, float a, float b) {
    switch (op) {
        case 1: return a + b;                   // add
        case 2: return a - b;                   // sub
        case 3: return a * b;                   // mul
        case 4: return a / b;                   // div
        case 5: return powf(a, b);              // pow
        case 6: return fmaxf(a, b);             // maximum
        case 7: return fminf(a, b);             // minimum
        case 8: return a > b ? 1.0f : 0.0f;     // gt mask
        case 9: return a == b ? 1.0f : 0.0f;    // eq mask
        default: return nanf("");
    }
}

extern "C" __global__ void unary_strided(
    const float* __restrict__ input,
    float* __restrict__ output,
    TensorMeta meta,
    unsigned int op,
    unsigned int numel)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numel) return;
    output[gid] = unary_eval(op, input[strided_offset(gid, meta)]);
}

extern "C" __global__ void binary_strided(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    TensorMeta ma,
    TensorMeta mb,
    unsigned int op,
    unsigned int numel)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numel) return;
    output[gid] = binary_eval(op, a[strided_offset(gid, ma)], b[strided_offset(gid, mb)]);
}

// ---- reductions ----------------------------------------------------------

__device__ __forceinline__ float reduce_identity(unsigned int op) {
    return op == 0 ? 0.0f : (op == 1 ? -INFINITY : INFINITY);
}

__device__ __forceinline__ float reduce_combine(unsigned int op, float acc, float x) {
    return op == 0 ? acc + x : (op == 1 ? fmaxf(acc, x) : fminf(acc, x));
}

// One thread per output element; walks the reduced subspace serially.
// `kept` maps an output index to the input base offset; `red` describes
// the reduced subspace (dims/strides only). A reduced extent of 0 leaves
// the identity in place.
extern "C" __global__ void reduce_axis(
    const float* __restrict__ input,
    float* __restrict__ output,
    TensorMeta kept,
    TensorMeta red,
    unsigned int op,          // 0 sum, 1 max, 2 min
    unsigned int out_numel)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= out_numel) return;
    int base = (int)strided_offset(gid, kept);
    float acc = reduce_identity(op);
    unsigned int red_numel = 1;
    for (unsigned int i = 0; i < red.rank; i++) red_numel *= red.dims[i];
    unsigned int coords[MAX_RANK] = {0};
    int off = base;
    for (unsigned int n = 0; n < red_numel; n++) {
        acc = reduce_combine(op, acc, input[(unsigned int)off]);
        for (unsigned int i = 0; i < red.rank; i++) {
            unsigned int d = red.rank - 1 - i;
            coords[d] += 1;
            off += red.strides[d];
            if (coords[d] < red.dims[d]) break;
            off -= (int)red.dims[d] * red.strides[d];
            coords[d] = 0;
        }
    }
    output[gid] = acc;
}

// Stage one of a full-tensor reduction: each 256-thread block reduces a
// grid-strided span through shared memory to one partial. Every
// __syncthreads() is at block scope — no barrier sits under a
// thread-dependent branch.
extern "C" __global__ void reduce_full_partials(
    const float* __restrict__ input,
    float* __restrict__ partials,
    TensorMeta meta,
    unsigned int op,
    unsigned int numel)
{
    __shared__ float shared[256];
    unsigned int lid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + lid;
    unsigned int grid_size = gridDim.x * blockDim.x;
    float acc = reduce_identity(op);
    for (unsigned int i = gid; i < numel; i += grid_size) {
        acc = reduce_combine(op, acc, input[strided_offset(i, meta)]);
    }
    shared[lid] = acc;
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] = reduce_combine(op, shared[lid], shared[lid + stride]);
        }
        __syncthreads();
    }
    if (lid == 0) partials[blockIdx.x] = shared[0];
}

// ---- matmul -----------------------------------------------------------------

// Tiled GEMM: 16x16 output tiles staged through shared memory.
// C[m, n] = A[m, k] x B[k, n]; one launch per batch element (base
// offsets passed in), inputs contiguous. Edge tiles load zeros, so the
// barriers are unconditional.
#define TILE 16

extern "C" __global__ void matmul_tiled(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    unsigned int m,
    unsigned int k,
    unsigned int n,
    unsigned int a_base,
    unsigned int b_base,
    unsigned int c_base)
{
    __shared__ float tile_a[TILE][TILE];
    __shared__ float tile_b[TILE][TILE];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int row = blockIdx.y * TILE + ty;
    unsigned int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;

    unsigned int tiles = (k + TILE - 1) / TILE;
    for (unsigned int t = 0; t < tiles; t++) {
        unsigned int ak = t * TILE + tx;
        unsigned int bk = t * TILE + ty;
        tile_a[ty][tx] = (row < m && ak < k) ? a[a_base + row * k + ak] : 0.0f;
        tile_b[ty][tx] = (bk < k && col < n) ? b[b_base + bk * n + col] : 0.0f;
        __syncthreads();
        for (unsigned int i = 0; i < TILE; i++) {
            acc += tile_a[ty][i] * tile_b[i][tx];
        }
        __syncthreads();
    }
    if (row < m && col < n) {
        c[c_base + row * n + col] = acc;
    }
}

// ---- optimizer ---------------------------------------------------------------

// Scalars of one fused Adam/AdamW step; layout shared with the host-side
// `AdamArgs` (10 four-byte words).
struct AdamArgs {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float bc1;                 // 1 - beta1^t
    float bc2;                 // 1 - beta2^t
    unsigned int decoupled;    // 1: AdamW (decay the weights), 0: Adam (L2 on the gradient)
    unsigned int has_state;    // 0 on the first step: m/v inputs are ignored
    unsigned int numel;
};

// The composite step's formula, one thread per element, three outputs.
// Inputs contiguous (the host makes them so). No barrier, no divergence
// question.
extern "C" __global__ void adam_step(
    const float* __restrict__ p_in,
    const float* __restrict__ g_in,
    const float* __restrict__ m_in,
    const float* __restrict__ v_in,
    float* __restrict__ p_out,
    float* __restrict__ m_out,
    float* __restrict__ v_out,
    AdamArgs a)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= a.numel) return;
    float p = p_in[gid];
    float g = g_in[gid];
    if (a.weight_decay != 0.0f) {
        if (a.decoupled != 0) p = p * (1.0f - a.lr * a.weight_decay);
        else g = g + p * a.weight_decay;
    }
    float m = a.has_state != 0 ? m_in[gid] * a.beta1 + g * (1.0f - a.beta1) : g * (1.0f - a.beta1);
    float g2 = g * g;
    float v = a.has_state != 0 ? v_in[gid] * a.beta2 + g2 * (1.0f - a.beta2) : g2 * (1.0f - a.beta2);
    float m_hat = m * (1.0f / a.bc1);
    float v_hat = v * (1.0f / a.bc2);
    float update = m_hat / (sqrtf(v_hat) + a.eps);
    p_out[gid] = p - update * a.lr;
    m_out[gid] = m;
    v_out[gid] = v;
}

// ---- gather / scatter ------------------------------------------------------

// index_select along `dim`: output is the source with dimension `dim`
// replaced by the selected rows, contiguous. Element gid of the output
// decomposes as (outer, k, inner); the source row is idx[k]. The source
// may be strided (meta), so nothing is copied first.
extern "C" __global__ void gather_dim(
    const float* __restrict__ input,
    const unsigned int* __restrict__ idx,
    float* __restrict__ output,
    TensorMeta meta,
    unsigned int dim,
    unsigned int idx_len,
    unsigned int out_numel)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= out_numel) return;
    unsigned int inner = 1;
    for (unsigned int d = dim + 1; d < meta.rank; d++) inner *= meta.dims[d];
    unsigned int nd = meta.dims[dim];
    unsigned int j = gid % inner;
    unsigned int k = (gid / inner) % idx_len;
    unsigned int o = gid / (inner * idx_len);
    unsigned int lin = (o * nd + idx[k]) * inner + j;
    output[gid] = input[strided_offset(lin, meta)];
}

// index_add along `dim`: output = a, then output[.., idx[k], ..] += src[.., k, ..]
// for every k. One thread per OUTPUT element scanning the index list:
// deterministic (adds in k order, as the CPU reference) and atomic-free,
// O(idx_len) per element. No barrier anywhere, so no convergence question.
extern "C" __global__ void scatter_add_dim(
    const float* __restrict__ a,
    const float* __restrict__ src,
    const unsigned int* __restrict__ idx,
    float* __restrict__ output,
    TensorMeta ma,
    TensorMeta ms,
    unsigned int dim,
    unsigned int idx_len,
    unsigned int numel)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numel) return;
    unsigned int inner = 1;
    for (unsigned int d = dim + 1; d < ma.rank; d++) inner *= ma.dims[d];
    unsigned int nd = ma.dims[dim];
    unsigned int j = gid % inner;
    unsigned int t = (gid / inner) % nd;
    unsigned int o = gid / (inner * nd);
    float acc = a[strided_offset(gid, ma)];
    for (unsigned int k = 0; k < idx_len; k++) {
        if (idx[k] == t) {
            unsigned int src_lin = (o * idx_len + k) * inner + j;
            acc += src[strided_offset(src_lin, ms)];
        }
    }
    output[gid] = acc;
}
