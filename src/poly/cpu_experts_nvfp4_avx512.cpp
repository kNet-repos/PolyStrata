// src/poly/cpu_experts_nvfp4_avx512.cpp - Custom Skylake-X+ AVX-512F/BW/DQ/VL NVFP4 CPU expert kernels.
//
// Compiled with: -mavx512f -mavx512bw -mavx512dq -mavx512vl -mf16c -mfma
// No namespace-scope dynamic initializers so non-AVX-512 CPUs never execute AVX-512 instructions at startup.
#include "cpu_experts_nvfp4_avx512.hpp"

#include <cstdint>

namespace {
constexpr int QK_NVFP4 = 64;
constexpr int QK_NVFP4_SUB = 16;
constexpr int QK8_0 = 32;

struct block_nvfp4 {
    uint8_t d[QK_NVFP4 / QK_NVFP4_SUB];
    uint8_t qs[QK_NVFP4 / 2];
};
static_assert(sizeof(block_nvfp4) == 36, "block_nvfp4 size mismatch");

struct block_q8_0 {
    uint16_t d;
    int8_t qs[QK8_0];
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 size mismatch");
} // namespace

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata::poly {
namespace {

constexpr float pow2i(int e) {
    float r = 1.0f;
    for (; e > 0; --e) r *= 2.0f;
    for (; e < 0; ++e) r *= 0.5f;
    return r;
}

// Compile-time UE4M3 table matching ggml_ue4m3_to_fp32 (halved for doubled E2M1 kvalues_fp4; 0x7F -> 0).
struct Ue4m3Table {
    float v[256];
    constexpr Ue4m3Table() : v() {
        for (int x = 0; x < 256; ++x) {
            if (x == 0 || x == 0x7F) {
                v[x] = 0.0f;
                continue;
            }
            const int e = (x >> 3) & 0xF;
            const int m = x & 0x7;
            const float raw = e == 0 ? (float) m * pow2i(-9) : (1.0f + (float) m / 8.0f) * pow2i(e - 7);
            v[x] = raw * 0.5f;
        }
    }
};
constexpr Ue4m3Table kUe4m3;

inline int prefetch_distance() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_NVFP4_PREFETCH");
        return e ? std::atoi(e) : 2048;
    }();
    return v;
}

inline float h2f(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h)));
}

inline uint16_t f2h(float f) {
    return (uint16_t) _mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT));
}

// Decode 32 bytes of block_nvfp4::qs into 64 4-bit codes in [sub0, sub2, sub1, sub3] 128-bit lane order.
// Zero cross-lane 256-bit permutes!
inline __m512i codes64_0213(const uint8_t* qs, const __m256i m4) {
    const __m256i q = _mm256_loadu_si256((const __m256i*) qs);
    const __m256i lo = _mm256_and_si256(q, m4);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_unpacklo_epi64(lo, hi)),
                              _mm256_unpackhi_epi64(lo, hi), 1);
}

// Build 16-lane scale vector [d0x4, d2x4, d1x4, d3x4] in a single vpermps.
inline __m512 sub_scales_0213(const uint8_t* d, const __m512i dup4_idx) {
    const __m128 s4 = _mm_setr_ps(kUe4m3.v[d[0]], kUe4m3.v[d[2]], kUe4m3.v[d[1]], kUe4m3.v[d[3]]);
    return _mm512_permutexvar_ps(dup4_idx, _mm512_castps128_ps512(s4));
}

// One 64-element block dot product for 1 row against pre-packed (q, dy).
inline __m512 dot_block_skx(const block_nvfp4& blk, const __m512i q, const __m512 dy,
                            const __m256i m4, const __m512i mag, const __m512i eight,
                            const __m512i zero, const __m512i ones, const __m512i dup4_idx,
                            const __m512 acc) {
    const __m512i codes = codes64_0213(blk.qs, m4);
    const __m512i g = _mm512_shuffle_epi8(mag, codes);
    const __mmask64 neg = _mm512_test_epi8_mask(codes, eight);
    const __m512i qs = _mm512_mask_sub_epi8(q, neg, zero, q);
    const __m512i p16 = _mm512_maddubs_epi16(g, qs);
    const __m512i p32 = _mm512_madd_epi16(p16, ones);
    const __m512 sw = sub_scales_0213(blk.d, dup4_idx);
    return _mm512_fmadd_ps(_mm512_mul_ps(sw, dy), _mm512_cvtepi32_ps(p32), acc);
}

// Quantize 32 floats into 16+16 int8 lanes and return FP16-rounded scale.
inline float quantize32_to_i8(const float* x, __m128i& q_lo16, __m128i& q_hi16) {
    const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
    const __m512 v0 = _mm512_loadu_ps(x);
    const __m512 v1 = _mm512_loadu_ps(x + 16);
    const __m512 vmax = _mm512_max_ps(_mm512_and_ps(v0, abs_mask), _mm512_and_ps(v1, abs_mask));
    const float amax = _mm512_reduce_max_ps(vmax);
    const float d = amax * (1.0f / 127.0f);
    const float id = d != 0.0f ? (1.0f / d) : 0.0f;
    const __m512 vid = _mm512_set1_ps(id);
    const __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(v0, vid));
    const __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(v1, vid));
    q_lo16 = _mm512_cvtepi32_epi8(i0);
    q_hi16 = _mm512_cvtepi32_epi8(i1);
    return h2f(f2h(d));
}

// Quantize 64 floats directly into one packed Nvfp4PackedActs512 block (q_dst: 64B aligned, dy_dst: 64B aligned).
inline void quantize64_and_pack_block(const float* x, uint8_t* q_dst, float* dy_dst) {
    __m128i q0, q1, q2, q3;
    const float d0 = quantize32_to_i8(x, q0, q1);
    const float d1 = quantize32_to_i8(x + 32, q2, q3);
    // [sub0=q0, sub2=q2, sub1=q1, sub3=q3]
    _mm_store_si128((__m128i*) (q_dst + 0), q0);
    _mm_store_si128((__m128i*) (q_dst + 16), q2);
    _mm_store_si128((__m128i*) (q_dst + 32), q1);
    _mm_store_si128((__m128i*) (q_dst + 48), q3);
    const __m512 dy = _mm512_mask_blend_ps((__mmask16) 0xF0F0, _mm512_set1_ps(d0), _mm512_set1_ps(d1));
    _mm512_store_ps(dy_dst, dy);
}

inline float apply_act(float g, float u, float gate_scale, float up_scale, float limit, bool gelu, SiTU situ) {
    g *= gate_scale;
    u *= up_scale;
    if (situ.beta > 0.f) {
        return situ.beta * std::tanh(g / situ.beta) / (1.f + std::exp(-g)) *
               (situ.linear_beta * std::tanh(u / situ.linear_beta));
    }
    if (gelu) {
        return 0.5f * g * (1.f + std::tanh(0.7978845608f * g * (1.f + 0.044715f * g * g))) * u;
    }
    g = std::min(g, limit);
    u = std::clamp(u, -limit, limit);
    return g / (1.f + std::exp(-g)) * u;
}

// NT == 1 specialized 4-row dot (2 gate rows + 2 up rows, or 4 down rows, sharing activation loads in ZMM!).
inline void dot4_rows_nt1(const uint8_t* r0, const uint8_t* r1, const uint8_t* r2, const uint8_t* r3,
                          const uint8_t* q_base, const float* dy_base, int nb, float out4[4]) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m512i mag = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12));
    const __m512i eight = _mm512_set1_epi8(8);
    const __m512i zero = _mm512_setzero_si512();
    const __m512i ones = _mm512_set1_epi16(1);
    const __m512i dup4_idx = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);

    __m512 a0 = _mm512_setzero_ps();
    __m512 a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps();
    __m512 a3 = _mm512_setzero_ps();

    const block_nvfp4* x0 = (const block_nvfp4*) r0;
    const block_nvfp4* x1 = (const block_nvfp4*) r1;
    const block_nvfp4* x2 = (const block_nvfp4*) r2;
    const block_nvfp4* x3 = (const block_nvfp4*) r3;
    const int pf = prefetch_distance();

    for (int ib = 0; ib < nb; ++ib) {
        if (pf > 0) {
            _mm_prefetch((const char*) (x0 + ib) + pf, _MM_HINT_T0);
            _mm_prefetch((const char*) (x1 + ib) + pf, _MM_HINT_T0);
            _mm_prefetch((const char*) (x2 + ib) + pf, _MM_HINT_T0);
            _mm_prefetch((const char*) (x3 + ib) + pf, _MM_HINT_T0);
        }
        const __m512i q = _mm512_load_si512((const __m512i*) (q_base + (size_t) ib * 64));
        const __m512 dy = _mm512_load_ps(dy_base + (size_t) ib * 16);
        a0 = dot_block_skx(x0[ib], q, dy, m4, mag, eight, zero, ones, dup4_idx, a0);
        a1 = dot_block_skx(x1[ib], q, dy, m4, mag, eight, zero, ones, dup4_idx, a1);
        a2 = dot_block_skx(x2[ib], q, dy, m4, mag, eight, zero, ones, dup4_idx, a2);
        a3 = dot_block_skx(x3[ib], q, dy, m4, mag, eight, zero, ones, dup4_idx, a3);
    }
    out4[0] = _mm512_reduce_add_ps(a0);
    out4[1] = _mm512_reduce_add_ps(a1);
    out4[2] = _mm512_reduce_add_ps(a2);
    out4[3] = _mm512_reduce_add_ps(a3);
}

// NT == 1 2-row dot (remainder or 1 gate + 1 up).
inline void dot2_rows_nt1(const uint8_t* r0, const uint8_t* r1,
                          const uint8_t* q_base, const float* dy_base, int nb, float out2[2]) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m512i mag = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12));
    const __m512i eight = _mm512_set1_epi8(8);
    const __m512i zero = _mm512_setzero_si512();
    const __m512i ones = _mm512_set1_epi16(1);
    const __m512i dup4_idx = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);

    __m512 a0 = _mm512_setzero_ps();
    __m512 a1 = _mm512_setzero_ps();
    const block_nvfp4* x0 = (const block_nvfp4*) r0;
    const block_nvfp4* x1 = (const block_nvfp4*) r1;
    const int pf = prefetch_distance();

    for (int ib = 0; ib < nb; ++ib) {
        if (pf > 0) {
            _mm_prefetch((const char*) (x0 + ib) + pf, _MM_HINT_T0);
            _mm_prefetch((const char*) (x1 + ib) + pf, _MM_HINT_T0);
        }
        const __m512i q = _mm512_load_si512((const __m512i*) (q_base + (size_t) ib * 64));
        const __m512 dy = _mm512_load_ps(dy_base + (size_t) ib * 16);
        a0 = dot_block_skx(x0[ib], q, dy, m4, mag, eight, zero, ones, dup4_idx, a0);
        a1 = dot_block_skx(x1[ib], q, dy, m4, mag, eight, zero, ones, dup4_idx, a1);
    }
    out2[0] = _mm512_reduce_add_ps(a0);
    out2[1] = _mm512_reduce_add_ps(a1);
}

inline float dot1_row_nt1(const uint8_t* r0, const uint8_t* q_base, const float* dy_base, int nb) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m512i mag = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12));
    const __m512i eight = _mm512_set1_epi8(8);
    const __m512i zero = _mm512_setzero_si512();
    const __m512i ones = _mm512_set1_epi16(1);
    const __m512i dup4_idx = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);

    __m512 a0 = _mm512_setzero_ps();
    const block_nvfp4* x0 = (const block_nvfp4*) r0;
    const int pf = prefetch_distance();
    for (int ib = 0; ib < nb; ++ib) {
        if (pf > 0) _mm_prefetch((const char*) (x0 + ib) + pf, _MM_HINT_T0);
        const __m512i q = _mm512_load_si512((const __m512i*) (q_base + (size_t) ib * 64));
        const __m512 dy = _mm512_load_ps(dy_base + (size_t) ib * 16);
        a0 = dot_block_skx(x0[ib], q, dy, m4, mag, eight, zero, ones, dup4_idx, a0);
    }
    return _mm512_reduce_add_ps(a0);
}

// Multi-token row dot (NT = 2..8): decodes each weight block ONCE and applies to all NT tokens!
template <int NT>
inline void row_dot_multi(const uint8_t* row, const uint8_t* const* q_ptrs, const float* const* dy_ptrs,
                          int nb, float res[NT]) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m512i mag = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12));
    const __m512i eight = _mm512_set1_epi8(8);
    const __m512i zero = _mm512_setzero_si512();
    const __m512i ones = _mm512_set1_epi16(1);
    const __m512i dup4_idx = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);

    __m512 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_ps();
    const block_nvfp4* x = (const block_nvfp4*) row;
    const int pf = prefetch_distance();

    for (int ib = 0; ib < nb; ++ib) {
        if (pf > 0) _mm_prefetch((const char*) (x + ib) + pf, _MM_HINT_T0);
        const __m512i codes = codes64_0213(x[ib].qs, m4);
        const __m512i g = _mm512_shuffle_epi8(mag, codes);
        const __mmask64 neg = _mm512_test_epi8_mask(codes, eight);
        const __m512 sw = sub_scales_0213(x[ib].d, dup4_idx);
        for (int t = 0; t < NT; ++t) {
            const __m512i q = _mm512_load_si512((const __m512i*) (q_ptrs[t] + (size_t) ib * 64));
            const __m512 dy = _mm512_load_ps(dy_ptrs[t] + (size_t) ib * 16);
            const __m512i qs = _mm512_mask_sub_epi8(q, neg, zero, q);
            const __m512i p16 = _mm512_maddubs_epi16(g, qs);
            const __m512i p32 = _mm512_madd_epi16(p16, ones);
            acc[t] = _mm512_fmadd_ps(_mm512_mul_ps(sw, dy), _mm512_cvtepi32_ps(p32), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(acc[t]);
}

// 2-row fused multi-token dot (e.g. gate[r] + up[r] or down[r] + down[r+1]):
// Loads each token's (q, dy) ONCE across both rows and overlaps memory latency on two weight streams.
template <int NT>
inline void row2_dot_multi(const uint8_t* row0, const uint8_t* row1,
                           const uint8_t* const* q_ptrs, const float* const* dy_ptrs,
                           int nb, float res0[NT], float res1[NT]) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m512i mag = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12));
    const __m512i eight = _mm512_set1_epi8(8);
    const __m512i zero = _mm512_setzero_si512();
    const __m512i ones = _mm512_set1_epi16(1);
    const __m512i dup4_idx = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);

    __m512 a0[NT], a1[NT];
    for (int t = 0; t < NT; ++t) {
        a0[t] = _mm512_setzero_ps();
        a1[t] = _mm512_setzero_ps();
    }
    const block_nvfp4* x0 = (const block_nvfp4*) row0;
    const block_nvfp4* x1 = (const block_nvfp4*) row1;
    const int pf = prefetch_distance();

    for (int ib = 0; ib < nb; ++ib) {
        if (pf > 0) {
            _mm_prefetch((const char*) (x0 + ib) + pf, _MM_HINT_T0);
            _mm_prefetch((const char*) (x1 + ib) + pf, _MM_HINT_T0);
        }
        const __m512i c0 = codes64_0213(x0[ib].qs, m4);
        const __m512i g0 = _mm512_shuffle_epi8(mag, c0);
        const __mmask64 n0 = _mm512_test_epi8_mask(c0, eight);
        const __m512 sw0 = sub_scales_0213(x0[ib].d, dup4_idx);

        const __m512i c1 = codes64_0213(x1[ib].qs, m4);
        const __m512i g1 = _mm512_shuffle_epi8(mag, c1);
        const __mmask64 n1 = _mm512_test_epi8_mask(c1, eight);
        const __m512 sw1 = sub_scales_0213(x1[ib].d, dup4_idx);

        for (int t = 0; t < NT; ++t) {
            const __m512i q = _mm512_load_si512((const __m512i*) (q_ptrs[t] + (size_t) ib * 64));
            const __m512 dy = _mm512_load_ps(dy_ptrs[t] + (size_t) ib * 16);

            const __m512i qs0 = _mm512_mask_sub_epi8(q, n0, zero, q);
            const __m512i p16_0 = _mm512_maddubs_epi16(g0, qs0);
            const __m512i p32_0 = _mm512_madd_epi16(p16_0, ones);
            a0[t] = _mm512_fmadd_ps(_mm512_mul_ps(sw0, dy), _mm512_cvtepi32_ps(p32_0), a0[t]);

            const __m512i qs1 = _mm512_mask_sub_epi8(q, n1, zero, q);
            const __m512i p16_1 = _mm512_maddubs_epi16(g1, qs1);
            const __m512i p32_1 = _mm512_madd_epi16(p16_1, ones);
            a1[t] = _mm512_fmadd_ps(_mm512_mul_ps(sw1, dy), _mm512_cvtepi32_ps(p32_1), a1[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        res0[t] = _mm512_reduce_add_ps(a0[t]);
        res1[t] = _mm512_reduce_add_ps(a1[t]);
    }
}

template <int NT>
void gu_rows_multi(const uint8_t* gate, const uint8_t* up, size_t row_bytes, int nb,
                   const uint8_t* const* q_ptrs, const float* const* dy_ptrs,
                   const int* exp_slots, float gate_scale, float up_scale, float limit, bool gelu, SiTU situ,
                   Nvfp4PackedActs512* fused_pack_dst, float* h_out, int n_ff, int r0, int r1) {
    if (fused_pack_dst && (r0 % 64 == 0) && ((r1 - r0) % 64 == 0)) {
        for (int rb = r0; rb < r1; rb += 64) {
            alignas(64) float hbuf[NT][64];
            for (int k = 0; k < 64; ++k) {
                const int r = rb + k;
                float g[NT], u[NT];
                if constexpr (NT <= 6) {
                    row2_dot_multi<NT>(gate + (size_t) r * row_bytes, up + (size_t) r * row_bytes,
                                       q_ptrs, dy_ptrs, nb, g, u);
                } else {
                    row_dot_multi<NT>(gate + (size_t) r * row_bytes, q_ptrs, dy_ptrs, nb, g);
                    row_dot_multi<NT>(up + (size_t) r * row_bytes, q_ptrs, dy_ptrs, nb, u);
                }
                for (int t = 0; t < NT; ++t) {
                    hbuf[t][k] = apply_act(g[t], u[t], gate_scale, up_scale, limit, gelu, situ);
                }
            }
            const int ib_dst = rb / 64;
            for (int t = 0; t < NT; ++t) {
                const size_t blk_idx = (size_t) exp_slots[t] * (size_t) fused_pack_dst->nb + (size_t) ib_dst;
                quantize64_and_pack_block(hbuf[t], fused_pack_dst->q_raw + blk_idx * 64,
                                          fused_pack_dst->dy_raw + blk_idx * 16);
            }
        }
        return;
    }
    for (int r = r0; r < r1; ++r) {
        float g[NT], u[NT];
        if constexpr (NT <= 6) {
            row2_dot_multi<NT>(gate + (size_t) r * row_bytes, up + (size_t) r * row_bytes,
                               q_ptrs, dy_ptrs, nb, g, u);
        } else {
            row_dot_multi<NT>(gate + (size_t) r * row_bytes, q_ptrs, dy_ptrs, nb, g);
            row_dot_multi<NT>(up + (size_t) r * row_bytes, q_ptrs, dy_ptrs, nb, u);
        }
        for (int t = 0; t < NT; ++t) {
            h_out[(size_t) exp_slots[t] * (size_t) n_ff + (size_t) r] =
                apply_act(g[t], u[t], gate_scale, up_scale, limit, gelu, situ);
        }
    }
}

template <int NT>
void down_expert_multi(const uint8_t* down, size_t d_row, int nb,
                       const uint8_t* const* q_ptrs, const float* const* dy_ptrs,
                       const int* exp_slots, float* scratch, int r_count, int n_experts, int r0) {
    int k = 0;
    if constexpr (NT <= 6) {
        for (; k + 1 < r_count; k += 2) {
            float res0[NT], res1[NT];
            row2_dot_multi<NT>(down + (size_t) (r0 + k) * d_row,
                               down + (size_t) (r0 + k + 1) * d_row,
                               q_ptrs, dy_ptrs, nb, res0, res1);
            float* s0 = scratch + (size_t) k * (size_t) n_experts;
            float* s1 = scratch + (size_t) (k + 1) * (size_t) n_experts;
            for (int t = 0; t < NT; ++t) {
                s0[exp_slots[t]] = res0[t];
                s1[exp_slots[t]] = res1[t];
            }
        }
    }
    for (; k < r_count; ++k) {
        float res[NT];
        row_dot_multi<NT>(down + (size_t) (r0 + k) * d_row, q_ptrs, dy_ptrs, nb, res);
        float* row_s = scratch + (size_t) k * (size_t) n_experts;
        for (int t = 0; t < NT; ++t) row_s[exp_slots[t]] = res[t];
    }
}

}  // namespace

Nvfp4PackedActs512::~Nvfp4PackedActs512() {
    std::free(q_raw);
    std::free(dy_raw);
}

void Nvfp4PackedActs512::prepare(int slots, int n_elems) {
    nb = n_elems / QK_NVFP4;
    n_slots = slots;
    const size_t need = (size_t) slots * (size_t) nb;
    if (need > cap_blocks) {
        std::free(q_raw);
        std::free(dy_raw);
        const size_t cap = std::max(need, size_t(64));
        q_raw = (uint8_t*) std::aligned_alloc(64, cap * 64);
        dy_raw = (float*) std::aligned_alloc(64, cap * 64);
        cap_blocks = cap;
    }
}

void quantize_row_q8_0_avx512(const float* x, void* vy, int n) {
    block_q8_0* y = (block_q8_0*) vy;
    const int nb = n / QK8_0;
    const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
    for (int i = 0; i < nb; ++i) {
        const float* xb = x + (size_t) i * 32;
        const __m512 v0 = _mm512_loadu_ps(xb);
        const __m512 v1 = _mm512_loadu_ps(xb + 16);
        const float amax = _mm512_reduce_max_ps(
            _mm512_max_ps(_mm512_and_ps(v0, abs_mask), _mm512_and_ps(v1, abs_mask)));
        const float d = amax * (1.0f / 127.0f);
        const float id = d != 0.0f ? (1.0f / d) : 0.0f;
        y[i].d = f2h(d);
        const __m512 vid = _mm512_set1_ps(id);
        const __m128i q0 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(v0, vid)));
        const __m128i q1 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(v1, vid)));
        _mm_storeu_si128((__m128i*) (y[i].qs + 0), q0);
        _mm_storeu_si128((__m128i*) (y[i].qs + 16), q1);
    }
}

void nvfp4_quantize_and_pack_f32_avx512(const float* x, int slot, int n, Nvfp4PackedActs512& dst) {
    const int nb = n / QK_NVFP4;
    const size_t base_blk = (size_t) slot * (size_t) nb;
    for (int ib = 0; ib < nb; ++ib) {
        quantize64_and_pack_block(x + (size_t) ib * 64,
                                  dst.q_raw + (base_blk + (size_t) ib) * 64,
                                  dst.dy_raw + (base_blk + (size_t) ib) * 16);
    }
}

void nvfp4_pack_q8_0_one_avx512(const uint8_t* q8_row, int slot, int n, Nvfp4PackedActs512& dst) {
    const int nb = n / QK_NVFP4;
    const block_q8_0* y = (const block_q8_0*) q8_row;
    const size_t base_blk = (size_t) slot * (size_t) nb;
    for (int ib = 0; ib < nb; ++ib) {
        const block_q8_0* yb = y + 2 * ib;
        uint8_t* q_dst = dst.q_raw + (base_blk + (size_t) ib) * 64;
        float* dy_dst = dst.dy_raw + (base_blk + (size_t) ib) * 16;
        // [sub0 = yb[0].qs[0..15], sub2 = yb[1].qs[0..15], sub1 = yb[0].qs[16..31], sub3 = yb[1].qs[16..31]]
        _mm_store_si128((__m128i*) (q_dst + 0), _mm_loadu_si128((const __m128i*) (yb[0].qs + 0)));
        _mm_store_si128((__m128i*) (q_dst + 16), _mm_loadu_si128((const __m128i*) (yb[1].qs + 0)));
        _mm_store_si128((__m128i*) (q_dst + 32), _mm_loadu_si128((const __m128i*) (yb[0].qs + 16)));
        _mm_store_si128((__m128i*) (q_dst + 48), _mm_loadu_si128((const __m128i*) (yb[1].qs + 16)));
        const float d0 = h2f(yb[0].d);
        const float d1 = h2f(yb[1].d);
        const __m512 dy = _mm512_mask_blend_ps((__mmask16) 0xF0F0, _mm512_set1_ps(d0), _mm512_set1_ps(d1));
        _mm512_store_ps(dy_dst, dy);
    }
}

void nvfp4_gu_rows_avx512(const uint8_t* gate, const uint8_t* up, size_t row_bytes, int n_embd,
                          const Nvfp4PackedActs512& in_acts, const int* tok_rows, const int* exp_slots, int nt,
                          float gate_scale, float up_scale, float limit, bool gelu, SiTU situ,
                          Nvfp4PackedActs512* fused_pack_dst, float* h_out, int n_ff, int r0, int r1) {
    const int nb = n_embd / QK_NVFP4;
    if (nt == 1) {
        const size_t a_blk = (size_t) tok_rows[0] * (size_t) nb;
        const uint8_t* q_base = in_acts.q_raw + a_blk * 64;
        const float* dy_base = in_acts.dy_raw + a_blk * 16;
        const int slot = exp_slots[0];
        if (fused_pack_dst && (r0 % 64 == 0) && ((r1 - r0) % 64 == 0)) {
            for (int rb = r0; rb < r1; rb += 64) {
                alignas(64) float hbuf[64];
                for (int k = 0; k < 64; k += 2) {
                    const int r = rb + k;
                    float gu4[4];
                    dot4_rows_nt1(gate + (size_t) r * row_bytes,
                                  gate + (size_t) (r + 1) * row_bytes,
                                  up + (size_t) r * row_bytes,
                                  up + (size_t) (r + 1) * row_bytes,
                                  q_base, dy_base, nb, gu4);
                    hbuf[k + 0] = apply_act(gu4[0], gu4[2], gate_scale, up_scale, limit, gelu, situ);
                    hbuf[k + 1] = apply_act(gu4[1], gu4[3], gate_scale, up_scale, limit, gelu, situ);
                }
                const size_t blk_idx = (size_t) slot * (size_t) fused_pack_dst->nb + (size_t) (rb / 64);
                quantize64_and_pack_block(hbuf, fused_pack_dst->q_raw + blk_idx * 64,
                                          fused_pack_dst->dy_raw + blk_idx * 16);
            }
            return;
        }
        float* h_dst = h_out + (size_t) slot * (size_t) n_ff;
        int r = r0;
        for (; r + 1 < r1; r += 2) {
            float gu4[4];
            dot4_rows_nt1(gate + (size_t) r * row_bytes,
                          gate + (size_t) (r + 1) * row_bytes,
                          up + (size_t) r * row_bytes,
                          up + (size_t) (r + 1) * row_bytes,
                          q_base, dy_base, nb, gu4);
            h_dst[r + 0] = apply_act(gu4[0], gu4[2], gate_scale, up_scale, limit, gelu, situ);
            h_dst[r + 1] = apply_act(gu4[1], gu4[3], gate_scale, up_scale, limit, gelu, situ);
        }
        for (; r < r1; ++r) {
            float gu2[2];
            dot2_rows_nt1(gate + (size_t) r * row_bytes, up + (size_t) r * row_bytes, q_base, dy_base, nb, gu2);
            h_dst[r] = apply_act(gu2[0], gu2[1], gate_scale, up_scale, limit, gelu, situ);
        }
        return;
    }

    // Multi-token path: slice into groups of up to 8 tokens to keep accumulators in ZMM registers.
    for (int t0 = 0; t0 < nt; t0 += 8) {
        const int w = std::min(8, nt - t0);
        const uint8_t* q_ptrs[8];
        const float* dy_ptrs[8];
        for (int i = 0; i < w; ++i) {
            const size_t a_blk = (size_t) tok_rows[t0 + i] * (size_t) nb;
            q_ptrs[i] = in_acts.q_raw + a_blk * 64;
            dy_ptrs[i] = in_acts.dy_raw + a_blk * 16;
        }
        const int* slots = exp_slots + t0;
        switch (w) {
            case 1:
                nvfp4_gu_rows_avx512(gate, up, row_bytes, n_embd, in_acts, tok_rows + t0, slots, 1,
                                     gate_scale, up_scale, limit, gelu, situ, fused_pack_dst, h_out, n_ff, r0, r1);
                break;
            case 2: gu_rows_multi<2>(gate, up, row_bytes, nb, q_ptrs, dy_ptrs, slots, gate_scale, up_scale, limit, gelu, situ, fused_pack_dst, h_out, n_ff, r0, r1); break;
            case 3: gu_rows_multi<3>(gate, up, row_bytes, nb, q_ptrs, dy_ptrs, slots, gate_scale, up_scale, limit, gelu, situ, fused_pack_dst, h_out, n_ff, r0, r1); break;
            case 4: gu_rows_multi<4>(gate, up, row_bytes, nb, q_ptrs, dy_ptrs, slots, gate_scale, up_scale, limit, gelu, situ, fused_pack_dst, h_out, n_ff, r0, r1); break;
            case 5: gu_rows_multi<5>(gate, up, row_bytes, nb, q_ptrs, dy_ptrs, slots, gate_scale, up_scale, limit, gelu, situ, fused_pack_dst, h_out, n_ff, r0, r1); break;
            case 6: gu_rows_multi<6>(gate, up, row_bytes, nb, q_ptrs, dy_ptrs, slots, gate_scale, up_scale, limit, gelu, situ, fused_pack_dst, h_out, n_ff, r0, r1); break;
            case 7: gu_rows_multi<7>(gate, up, row_bytes, nb, q_ptrs, dy_ptrs, slots, gate_scale, up_scale, limit, gelu, situ, fused_pack_dst, h_out, n_ff, r0, r1); break;
            default: gu_rows_multi<8>(gate, up, row_bytes, nb, q_ptrs, dy_ptrs, slots, gate_scale, up_scale, limit, gelu, situ, fused_pack_dst, h_out, n_ff, r0, r1); break;
        }
    }
}

void nvfp4_down_rows_avx512(const CpuExpert* experts, const int* by, const int* run, int n_run,
                            int n_experts, int rows, size_t d_row, int n_ff, int n_embd,
                            const Nvfp4PackedActs512& h_acts, float* out, int r0, int r1) {
    const int nb = n_ff / QK_NVFP4;
    const int r_count = r1 - r0;

    // Fast path for single-token decode (rows == 1): expert-outer / row-inner so each expert's down slice
    // [r0..r1) is one contiguous ~20 KB sequential stream and h_acts[j] stays pinned in L1d!
    // Exact same expert accumulation order (j = 0..n_experts-1) preserves bit-exact parity.
    if (rows == 1 && r_count <= 64) {
        alignas(64) float acc[64];
        for (int k = 0; k < r_count; ++k) acc[k] = 0.f;
        for (int j = 0; j < n_experts; ++j) {
            const uint8_t* down = experts[j].down;
            const size_t a_blk = (size_t) j * (size_t) nb;
            const uint8_t* q_base = h_acts.q_raw + a_blk * 64;
            const float* dy_base = h_acts.dy_raw + a_blk * 16;
            const float w = experts[j].weight;
            int k = 0;
            for (; k + 3 < r_count; k += 4) {
                const int r = r0 + k;
                float s4[4];
                dot4_rows_nt1(down + (size_t) (r + 0) * d_row,
                              down + (size_t) (r + 1) * d_row,
                              down + (size_t) (r + 2) * d_row,
                              down + (size_t) (r + 3) * d_row,
                              q_base, dy_base, nb, s4);
                acc[k + 0] += w * s4[0];
                acc[k + 1] += w * s4[1];
                acc[k + 2] += w * s4[2];
                acc[k + 3] += w * s4[3];
            }
            for (; k < r_count; ++k) {
                const float s = dot1_row_nt1(down + (size_t) (r0 + k) * d_row, q_base, dy_base, nb);
                acc[k] += w * s;
            }
        }
        for (int k = 0; k < r_count; ++k) out[r0 + k] = acc[k];
        return;
    }

    // Multi-token path (rows > 1, e.g. DSpark verify or prefill CPU share): reuse each expert's down
    // weights across all tokens that picked it, then sum per token in exact original expert order.
    static thread_local std::vector<float> scratch;
    scratch.resize((size_t) r_count * (size_t) n_experts);
    float* sbuf = scratch.data();

    for (int u = 0; u < n_run; ++u) {
        const int i0 = run[u], i1 = run[u + 1];
        const int nt = i1 - i0;
        const uint8_t* down = experts[by[i0]].down;
        if (nt == 1) {
            const int j = by[i0];
            const size_t a_blk = (size_t) j * (size_t) nb;
            const uint8_t* q_base = h_acts.q_raw + a_blk * 64;
            const float* dy_base = h_acts.dy_raw + a_blk * 16;
            int k = 0;
            for (; k + 3 < r_count; k += 4) {
                float s4[4];
                dot4_rows_nt1(down + (size_t) (r0 + k + 0) * d_row,
                              down + (size_t) (r0 + k + 1) * d_row,
                              down + (size_t) (r0 + k + 2) * d_row,
                              down + (size_t) (r0 + k + 3) * d_row,
                              q_base, dy_base, nb, s4);
                sbuf[(size_t) (k + 0) * (size_t) n_experts + (size_t) j] = s4[0];
                sbuf[(size_t) (k + 1) * (size_t) n_experts + (size_t) j] = s4[1];
                sbuf[(size_t) (k + 2) * (size_t) n_experts + (size_t) j] = s4[2];
                sbuf[(size_t) (k + 3) * (size_t) n_experts + (size_t) j] = s4[3];
            }
            for (; k < r_count; ++k) {
                sbuf[(size_t) k * (size_t) n_experts + (size_t) j] =
                    dot1_row_nt1(down + (size_t) (r0 + k) * d_row, q_base, dy_base, nb);
            }
            continue;
        }
        for (int t0 = 0; t0 < nt; t0 += 8) {
            const int w = std::min(8, nt - t0);
            const uint8_t* q_ptrs[8];
            const float* dy_ptrs[8];
            int slots[8];
            for (int i = 0; i < w; ++i) {
                const int j = by[i0 + t0 + i];
                slots[i] = j;
                const size_t a_blk = (size_t) j * (size_t) nb;
                q_ptrs[i] = h_acts.q_raw + a_blk * 64;
                dy_ptrs[i] = h_acts.dy_raw + a_blk * 16;
            }
            switch (w) {
                case 1:
                    for (int k = 0; k < r_count; ++k)
                        sbuf[(size_t) k * (size_t) n_experts + (size_t) slots[0]] =
                            dot1_row_nt1(down + (size_t) (r0 + k) * d_row, q_ptrs[0], dy_ptrs[0], nb);
                    break;
                case 2: down_expert_multi<2>(down, d_row, nb, q_ptrs, dy_ptrs, slots, sbuf, r_count, n_experts, r0); break;
                case 3: down_expert_multi<3>(down, d_row, nb, q_ptrs, dy_ptrs, slots, sbuf, r_count, n_experts, r0); break;
                case 4: down_expert_multi<4>(down, d_row, nb, q_ptrs, dy_ptrs, slots, sbuf, r_count, n_experts, r0); break;
                case 5: down_expert_multi<5>(down, d_row, nb, q_ptrs, dy_ptrs, slots, sbuf, r_count, n_experts, r0); break;
                case 6: down_expert_multi<6>(down, d_row, nb, q_ptrs, dy_ptrs, slots, sbuf, r_count, n_experts, r0); break;
                case 7: down_expert_multi<7>(down, d_row, nb, q_ptrs, dy_ptrs, slots, sbuf, r_count, n_experts, r0); break;
                default: down_expert_multi<8>(down, d_row, nb, q_ptrs, dy_ptrs, slots, sbuf, r_count, n_experts, r0); break;
            }
        }
    }

    for (int k = 0; k < r_count; ++k) {
        const float* s = sbuf + (size_t) k * (size_t) n_experts;
        int j = 0;
        for (int row = 0; row < rows; ++row) {
            float acc = 0.f;
            for (; j < n_experts && experts[j].row == row; ++j) acc += experts[j].weight * s[j];
            out[(size_t) row * (size_t) n_embd + (size_t) (r0 + k)] = acc;
        }
    }
}


struct block_q8_K_local {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};
static_assert(sizeof(block_q8_K_local) == 292, "block_q8_K size mismatch");

void quantize_row_q8_K_avx512(const float* x, void* vy, int n) {
    auto* y = (block_q8_K_local*) vy;
    const int nb = n / 256;
    const __m512i v_min = _mm512_set1_epi32(-127);
    const __m512i v_max = _mm512_set1_epi32(127);
    const __m512 sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));

    for (int i = 0; i < nb; ++i) {
        const float* xi = x + (size_t) i * 256;
        __m512 v[16];
        __m512 vmax0 = _mm512_setzero_ps(), vmax1 = _mm512_setzero_ps();
        for (int g = 0; g < 16; g += 2) {
            v[g] = _mm512_loadu_ps(xi + g * 16);
            v[g + 1] = _mm512_loadu_ps(xi + (g + 1) * 16);
            vmax0 = _mm512_max_ps(vmax0, _mm512_and_ps(v[g], sign_mask));
            vmax1 = _mm512_max_ps(vmax1, _mm512_and_ps(v[g + 1], sign_mask));
        }
        const float amax = _mm512_reduce_max_ps(_mm512_max_ps(vmax0, vmax1));
        if (amax == 0.f) {
            y[i].d = 0.f;
            std::memset(y[i].qs, 0, sizeof(y[i].qs));
            std::memset(y[i].bsums, 0, sizeof(y[i].bsums));
            continue;
        }
        const __m512 vamax = _mm512_set1_ps(amax);
        float max_signed = amax;
        for (int g = 0; g < 16; ++g) {
            const __mmask16 m = _mm512_cmp_ps_mask(_mm512_and_ps(v[g], sign_mask), vamax, _CMP_EQ_OQ);
            if (m) {
                max_signed = xi[g * 16 + __builtin_ctz((unsigned) m)];
                break;
            }
        }
        const float iscale = -127.f / max_signed;
        y[i].d = 1.f / iscale;
        const __m512 visc = _mm512_set1_ps(iscale);
        for (int g = 0; g < 16; ++g) {
            __m512i q32 = _mm512_cvtps_epi32(_mm512_mul_ps(v[g], visc));
            q32 = _mm512_min_epi32(_mm512_max_epi32(q32, v_min), v_max);
            _mm_storeu_si128((__m128i*) (y[i].qs + g * 16), _mm512_cvtepi32_epi8(q32));
            y[i].bsums[g] = (int16_t) _mm512_reduce_add_epi32(q32);
        }
    }
}

}  // namespace strata::poly
