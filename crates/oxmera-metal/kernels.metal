// oxmera Metal kernels: strided elementwise, axis reductions, threadgroup
// full reduction, and tiled matmul. f32 throughout; strided access is
// described by TensorMeta (dims/strides/offset up to rank 8), with
// stride 0 encoding broadcast dimensions.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint MAX_RANK = 8;

struct TensorMeta {
    uint rank;
    uint offset;
    uint dims[MAX_RANK];
    int strides[MAX_RANK];
};

// Storage offset of logical element `gid` (row-major over `out_dims`).
static inline uint strided_offset(uint gid, constant TensorMeta &m) {
    uint rem = gid;
    int off = int(m.offset);
    for (uint i = 0; i < m.rank; i++) {
        uint d = m.rank - 1 - i;
        uint dim = m.dims[d];
        uint coord = rem % dim;
        rem /= dim;
        off += int(coord) * m.strides[d];
    }
    return uint(off);
}

// ---- elementwise -----------------------------------------------------------

static inline float unary_eval(uint op, float x) {
    switch (op) {
        case 0: return x;                    // copy / contiguous
        case 1: return -x;                   // neg
        case 2: return exp(x);               // exp
        case 3: return log(x);               // ln
        case 4: return fabs(x);              // abs
        case 5: return sqrt(x);              // sqrt
        case 6: return sin(x);               // sin
        case 7: return cos(x);               // cos
        case 8: return tanh(x);              // tanh
        case 9: return fmax(x, 0.0f);        // relu
        case 10: {                           // gelu (tanh approximation)
            float u = 0.7978845608f * (x + 0.044715f * x * x * x);
            return 0.5f * x * (1.0f + tanh(u));
        }
        case 11: return 1.0f / (1.0f + exp(-x)); // sigmoid
        default: return NAN;
    }
}

static inline float binary_eval(uint op, float a, float b) {
    switch (op) {
        case 1: return a + b;                 // add
        case 2: return a - b;                 // sub
        case 3: return a * b;                 // mul
        case 4: return a / b;                 // div
        case 5: return pow(a, b);             // pow
        case 6: return fmax(a, b);            // maximum
        case 7: return fmin(a, b);            // minimum
        case 8: return a > b ? 1.0f : 0.0f;   // gt mask
        case 9: return a == b ? 1.0f : 0.0f;  // eq mask
        default: return NAN;
    }
}

kernel void unary_strided(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant TensorMeta &meta [[buffer(2)]],
    constant uint &op [[buffer(3)]],
    constant uint &numel [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numel) return;
    output[gid] = unary_eval(op, input[strided_offset(gid, meta)]);
}

kernel void binary_strided(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant TensorMeta &ma [[buffer(3)]],
    constant TensorMeta &mb [[buffer(4)]],
    constant uint &op [[buffer(5)]],
    constant uint &numel [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numel) return;
    output[gid] = binary_eval(op, a[strided_offset(gid, ma)], b[strided_offset(gid, mb)]);
}

// ---- reductions ----------------------------------------------------------

// One thread per output element; walks the reduced subspace serially.
// `kept` describes the mapping from output index to input base offset;
// `red` describes the reduced subspace (dims/strides only).
kernel void reduce_axis(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant TensorMeta &kept [[buffer(2)]],
    constant TensorMeta &red [[buffer(3)]],
    constant uint &op [[buffer(4)]],       // 0 sum, 1 max, 2 min
    constant uint &out_numel [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= out_numel) return;
    int base = int(strided_offset(gid, kept));
    float acc = op == 0 ? 0.0f : (op == 1 ? -INFINITY : INFINITY);
    uint red_numel = 1;
    for (uint i = 0; i < red.rank; i++) red_numel *= red.dims[i];
    uint coords[MAX_RANK] = {0};
    int off = base;
    for (uint n = 0; n < red_numel; n++) {
        float x = input[uint(off)];
        acc = op == 0 ? acc + x : (op == 1 ? fmax(acc, x) : fmin(acc, x));
        for (uint i = 0; i < red.rank; i++) {
            uint d = red.rank - 1 - i;
            coords[d] += 1;
            off += red.strides[d];
            if (coords[d] < red.dims[d]) break;
            off -= int(red.dims[d]) * red.strides[d];
            coords[d] = 0;
        }
    }
    output[gid] = acc;
}

// Stage one of a full-tensor reduction: each threadgroup reduces a
// grid-strided span through threadgroup memory to one partial.
kernel void reduce_full_partials(
    device const float *input [[buffer(0)]],
    device float *partials [[buffer(1)]],
    constant TensorMeta &meta [[buffer(2)]],
    constant uint &op [[buffer(3)]],
    constant uint &numel [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]],
    uint grid_size [[threads_per_grid]])
{
    threadgroup float shared[256];
    float acc = op == 0 ? 0.0f : (op == 1 ? -INFINITY : INFINITY);
    for (uint i = gid; i < numel; i += grid_size) {
        float x = input[strided_offset(i, meta)];
        acc = op == 0 ? acc + x : (op == 1 ? fmax(acc, x) : fmin(acc, x));
    }
    shared[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) {
            float other = shared[lid + stride];
            shared[lid] = op == 0 ? shared[lid] + other
                        : (op == 1 ? fmax(shared[lid], other) : fmin(shared[lid], other));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) partials[tg] = shared[0];
}

// ---- matmul -----------------------------------------------------------------

// Tiled GEMM: 16x16 output tiles staged through threadgroup memory.
// C[m, n] = A[m, k] x B[k, n]; one kernel per batch element (base offsets
// passed in), inputs contiguous.
constant constexpr uint TILE = 16;

kernel void matmul_tiled(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant uint &m [[buffer(3)]],
    constant uint &k [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    constant uint &a_base [[buffer(6)]],
    constant uint &b_base [[buffer(7)]],
    constant uint &c_base [[buffer(8)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]])
{
    threadgroup float tile_a[TILE][TILE];
    threadgroup float tile_b[TILE][TILE];

    uint row = gid.y;
    uint col = gid.x;
    float acc = 0.0f;

    uint tiles = (k + TILE - 1) / TILE;
    for (uint t = 0; t < tiles; t++) {
        uint ak = t * TILE + tid.x;
        uint bk = t * TILE + tid.y;
        tile_a[tid.y][tid.x] = (row < m && ak < k) ? a[a_base + row * k + ak] : 0.0f;
        tile_b[tid.y][tid.x] = (bk < k && col < n) ? b[b_base + bk * n + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < TILE; i++) {
            acc += tile_a[tid.y][i] * tile_b[i][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < m && col < n) {
        c[c_base + row * n + col] = acc;
    }
}
