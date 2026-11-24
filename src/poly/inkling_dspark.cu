// The DSpark drafter's own launches (inkling_dspark.hpp).
#include "inkling_dspark.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>

namespace strata::poly::dspark {
namespace {

__host__ __device__ int div_up(int64_t a, int b) { return (int) ((a + b - 1) / b); }

template <int Threads> __device__ float block_sum(float x, float* waves) {
    for (int d = 16; d > 0; d /= 2) x += __shfl_down(x, d, 32);
    const int lane = threadIdx.x & 31, wave = threadIdx.x / 32;
    if (lane == 0) waves[wave] = x;
    __syncthreads();
    float total = 0.f;
    for (int i = 0; i < Threads / 32; ++i) total += waves[i];
    __syncthreads();
    return total;
}

__global__ void mix_taps_kernel(Taps t, const float* norm, float* out, int first, float eps) {
    __shared__ float waves[8];
    const size_t at = (size_t) (first + blockIdx.x) * kWidth;
    float* y = out + (size_t) blockIdx.x * kWidth;
    float ss = 0.f;
    for (int i = threadIdx.x; i < kWidth; i += 256) {
        float x = 0.f;
        for (int j = 0; j < kTaps; ++j) x = __fmaf_rn(t.w[j], t.tap[j][at + i], x);
        y[i] = x;
        ss = __fmaf_rn(x, x, ss);
    }
    const float s = rsqrtf(block_sum<256>(ss, waves) / kWidth + eps);
    for (int i = threadIdx.x; i < kWidth; i += 256) y[i] = y[i] * s * norm[i];
}

// The attention: a block takes one K/V group's 24 queries (8 rows x 3 heads) over tiles of 32 keys, tile b, b + parts,
// ..., with an online softmax; a second launch adds the parts.  A thread is a value of the head (128 of them).
constexpr int kQ = kBlock * (kHeads / kKv), kTile = 32, kParts = 64;
constexpr float kScale = 0.08838834764831845f;   // 1 / sqrt(128)

// Row j of a tile from its first key `first`: a cached position (slot0 + j, wrapped at ring) or a block row.
__device__ __forceinline__ float key_value(const Q8Rows& c, const float* own, int64_t kk, int64_t n_ctx, int64_t slot,
                                           int g, int d) {
    if (kk < n_ctx) {
        const int i = g * kHead + d;
        return (float) c.codes[slot * kKey + i] * __half2float(__ushort_as_half(c.scales[slot * (kKey / 32) + i / 32]));
    }
    return own[((kk - n_ctx) * kKv + g) * kHead + d];
}

__global__ void __launch_bounds__(kHead) attend_part_kernel(const float* q, const float* kb, const float* vb, Q8Rows k,
                                                             Q8Rows v, int64_t lo, int64_t n_ctx, int ring, int parts,
                                                             float* part) {
    __shared__ float qs[kQ][kHead];
    __shared__ float tile[kTile][kHead + 1];
    __shared__ float s[kQ][kTile];
    __shared__ float m[kQ], l[kQ], fac[kQ];
    const int g = blockIdx.x, b = blockIdx.y, d = threadIdx.x, lane = d % 32, wave = d / 32;
    for (int qi = 0; qi < kQ; ++qi) {
        const int row = qi / 3, head = g * 3 + qi % 3;
        qs[qi][d] = q[((size_t) row * kHeads + head) * kHead + d];
    }
    if (d < kQ) {
        m[d] = -INFINITY;
        l[d] = 0.f;
    }
    float acc[kQ];
    for (int qi = 0; qi < kQ; ++qi) acc[qi] = 0.f;
    const int64_t keys = n_ctx + kBlock;
    const int tiles = div_up(keys, kTile);
    for (int t = b; t < tiles; t += parts) {
        const int64_t first = (int64_t) t * kTile;
        const int count = (int) std::min<int64_t>(kTile, keys - first);
        // the slots of the tile's cached keys: consecutive positions, wrapped once at most
        int64_t slot0 = lo + first;
        if (ring) slot0 %= ring;
        __syncthreads();
        for (int j = 0; j < count; ++j) {
            int64_t slot = slot0 + j;
            if (ring && slot >= ring) slot -= ring;
            tile[j][d] = key_value(k, kb, first + j, n_ctx, slot, g, d);
        }
        __syncthreads();
        // a lane a key, a wave six of the queries
        {
            const int j = lane;
            float dot[kQ / 4] = {};
            if (j < count)
                for (int e = 0; e < kHead; ++e) {
                    const float kv = tile[j][e];
#pragma unroll
                    for (int i = 0; i < kQ / 4; ++i) dot[i] = __fmaf_rn(qs[wave * (kQ / 4) + i][e], kv, dot[i]);
                }
#pragma unroll
            for (int i = 0; i < kQ / 4; ++i) s[wave * (kQ / 4) + i][j] = j < count ? dot[i] * kScale : -INFINITY;
        }
        __syncthreads();
        if (d < kQ) {
            float mx = m[d];
            for (int j = 0; j < count; ++j) mx = fmaxf(mx, s[d][j]);
            const float f = expf(m[d] - mx);
            float sum = 0.f;
            for (int j = 0; j < kTile; ++j) {
                const float p = j < count ? expf(s[d][j] - mx) : 0.f;
                s[d][j] = p;
                sum += p;
            }
            m[d] = mx;
            l[d] = l[d] * f + sum;
            fac[d] = f;
        }
        __syncthreads();
        for (int j = 0; j < count; ++j) {
            int64_t slot = slot0 + j;
            if (ring && slot >= ring) slot -= ring;
            tile[j][d] = key_value(v, vb, first + j, n_ctx, slot, g, d);
        }
        __syncthreads();
        for (int qi = 0; qi < kQ; ++qi) {
            float a = acc[qi] * fac[qi];
            for (int j = 0; j < count; ++j) a = __fmaf_rn(s[qi][j], tile[j][d], a);
            acc[qi] = a;
        }
    }
    __syncthreads();
    float* out = part + ((size_t) g * parts + b) * kQ * (kHead + 2);
    for (int qi = 0; qi < kQ; ++qi) out[qi * (kHead + 2) + d] = acc[qi];
    if (d < kQ) {
        out[d * (kHead + 2) + kHead] = m[d];
        out[d * (kHead + 2) + kHead + 1] = l[d];
    }
}

__global__ void attend_add_kernel(const float* part, int parts, float* out) {
    const int g = blockIdx.x, qi = blockIdx.y, d = threadIdx.x;
    float mx = -INFINITY;
    for (int p = 0; p < parts; ++p) mx = fmaxf(mx, part[(((size_t) g * parts + p) * kQ + qi) * (kHead + 2) + kHead]);
    float sum = 0.f, acc = 0.f;
    for (int p = 0; p < parts; ++p) {
        const float* at = part + (((size_t) g * parts + p) * kQ + qi) * (kHead + 2);
        const float f = expf(at[kHead] - mx);
        sum += at[kHead + 1] * f;
        acc += at[d] * f;
    }
    const int row = qi / 3, head = g * 3 + qi % 3;
    out[((size_t) row * kHeads + head) * kHead + d] = acc / sum;
}

__global__ void joint_kernel(const float* state, const float* w1, const float* hidden, const float* context,
                             float* x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kRank) x[i] = state[i];
    else if (i < 2 * kRank) x[i] = w1[i - kRank];
    else if (i < 2 * kRank + kWidth) x[i] = hidden[i - 2 * kRank];
    else if (i < 2 * kRank + 2 * kWidth) x[i] = context[i - 2 * kRank - kWidth];
}

__global__ void recur_kernel(const float* j, const float* bias, float* state, float* latent) {
    const int i = threadIdx.x;
    const float gate = 1.f / (1.f + expf(-(j[i] + bias[i])));
    const float cand = tanhf(j[kRank + i] + bias[kRank + i]);
    state[i] = cand + gate * (state[i] - cand);
    latent[i] = tanhf(j[2 * kRank + i] + bias[2 * kRank + i]);
}

// Custom RDNA2 wave32 coalesced 2-vocab-per-thread DSpark head projection:
// 32-bit packed loads (bf16x2 or half2) halve LDS/loop overhead and saturate 128B cache-line bus.
template <bool kBf16>
__global__ void __launch_bounds__(kRank)
head_kernel_vec2(const float* __restrict__ base, const float* __restrict__ bias2, const uint32_t* __restrict__ w2_pair,
                 const float* __restrict__ latent, float* __restrict__ logits,
                 int n_active, int n_total, int ld_pairs, float cap) {
    __shared__ float lat[kRank];
    lat[threadIdx.x] = latent[threadIdx.x];
    __syncthreads();
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int v0 = pair_idx * 2;
    if (v0 >= n_total) return;
    if (v0 >= n_active) {
        logits[v0] = -INFINITY;
        if (v0 + 1 < n_total) logits[v0 + 1] = -INFINITY;
        return;
    }
    float acc0 = 0.f, acc1 = 0.f;
    const uint32_t* wp = w2_pair + pair_idx;
#pragma unroll 8
    for (int i = 0; i < kRank; ++i) {
        const uint32_t p = wp[(size_t) i * (size_t) ld_pairs];
        const float l = lat[i];
        float w0, w1;
        if constexpr (kBf16) {
            w0 = __uint_as_float(p << 16);
            w1 = __uint_as_float(p & 0xFFFF0000u);
        } else {
            const float2 f2 = __half22float2(__halves2half2(__ushort_as_half((uint16_t) p),
                                                            __ushort_as_half((uint16_t) (p >> 16))));
            w0 = f2.x;
            w1 = f2.y;
        }
        acc0 = __fmaf_rn(l, w0, acc0);
        acc1 = __fmaf_rn(l, w1, acc1);
    }
    const float inv_cap = 1.f / cap;
    logits[v0] = cap * tanhf((base[v0] + bias2[v0] + acc0) * inv_cap);
    if (v0 + 1 < n_active) {
        logits[v0 + 1] = cap * tanhf((base[v0 + 1] + bias2[v0 + 1] + acc1) * inv_cap);
    } else if (v0 + 1 < n_total) {
        logits[v0 + 1] = -INFINITY;
    }
}

__global__ void fill_rows_kernel(const float* src, float* dst, int width, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * rows) dst[i] = src[i % width];
}

__global__ void to_half_kernel(const float* src, uint16_t* dst, int n, int rows, int64_t ld) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) n * rows) return;
    const int64_t r = i / n, c = i % n;
    dst[r * ld + c] = __half_as_ushort(__float2half(src[i]));
}

}  // namespace

void mix_taps(const Taps& t, const float* norm, float* out, int first, int n, float eps, cudaStream_t st) {
    if (n > 0) mix_taps_kernel<<<n, 256, 0, st>>>(t, norm, out, first, eps);
}

size_t attend_floats() { return (size_t) kKv * kParts * kQ * (kHead + 2); }

void attend(const float* q, const float* kb, const float* vb, Q8Rows k, Q8Rows v, int64_t lo, int64_t pos, int ring,
            float* scratch, float* out, cudaStream_t st) {
    const int64_t n_ctx = pos - lo;
    const int parts = std::min(kParts, div_up(n_ctx + kBlock, kTile));
    attend_part_kernel<<<dim3(kKv, parts), kHead, 0, st>>>(q, kb, vb, k, v, lo, n_ctx, ring, parts, scratch);
    attend_add_kernel<<<dim3(kKv, kQ), kHead, 0, st>>>(scratch, parts, out);
}

void joint(const float* state, const float* w1, const float* hidden, const float* context, float* x,
           cudaStream_t st) {
    joint_kernel<<<div_up(2 * kRank + 2 * kWidth, 256), 256, 0, st>>>(state, w1, hidden, context, x);
}

void recur(const float* j, const float* bias, float* state, float* latent, cudaStream_t st) {
    recur_kernel<<<1, kRank, 0, st>>>(j, bias, state, latent);
}

void head(const float* base, const float* bias2, const uint16_t* w2, const float* latent, float* logits,
          int n_active, int n_total, int ld, bool w2_bf16, float cap, cudaStream_t st) {
    const int n_pairs = (n_total + 1) / 2;
    const int ld_pairs = ld / 2;
    const dim3 grid(div_up(n_pairs, kRank));
    if (w2_bf16) {
        head_kernel_vec2<true><<<grid, kRank, 0, st>>>(base, bias2, (const uint32_t*) w2, latent, logits,
                                                       n_active, n_total, ld_pairs, cap);
    } else {
        head_kernel_vec2<false><<<grid, kRank, 0, st>>>(base, bias2, (const uint32_t*) w2, latent, logits,
                                                        n_active, n_total, ld_pairs, cap);
    }
}

void fill_rows(const float* src, float* dst, int width, int rows, cudaStream_t st) {
    if (rows > 0) fill_rows_kernel<<<div_up((int64_t) width * rows, 256), 256, 0, st>>>(src, dst, width, rows);
}

void to_half(const float* src, uint16_t* dst, int n, int rows, int64_t ld, cudaStream_t st) {
    to_half_kernel<<<div_up((int64_t) n * rows, 256), 256, 0, st>>>(src, dst, n, rows, ld);
}

}  // namespace strata::poly::dspark
