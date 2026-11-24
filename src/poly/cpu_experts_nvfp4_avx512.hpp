// src/poly/cpu_experts_nvfp4_avx512.hpp - Custom Skylake-X+ AVX-512F/BW/DQ/VL NVFP4 CPU expert kernels.
//
// Works on Intel Skylake-X / Xeon W-2135 (AVX-512F + BW + DQ + VL + F16C + FMA) without requiring
// AVX512-VNNI or AVX512-VBMI. Pre-packs Q8_0 activations in [sub0, sub2, sub1, sub3] 64-byte lane order
// so NVFP4 nibble decoding uses zero cross-lane 256-bit permutes in the hot inner loop, fuses gate+up
// with 4-row activation reuse (NT=1) or multi-token weight reuse (NT=2..8), and fuses inline SiTU +
// Q8_0 quantization + pre-packing directly into the down-projection activation buffer.
#pragma once

#include "cpu_experts.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::poly {

// Runtime CPUID check (implemented in cpu_experts.cpp, outside the AVX-512 TU).
bool nvfp4_avx512_skx_supported() noexcept;

// Fast AVX-512F Q8_0 quantizer (32-element blocks).
void quantize_row_q8_0_avx512(const float* x, void* vy, int n);

// 64-byte aligned packed Q8_0 activations in [sub0, sub2, sub1, sub3] order for zero-permute NVFP4 dot.
struct Nvfp4PackedActs512 {
    static constexpr int kMaxBlocks = 64;   // up to 4096 elements per row
    int nb = 0;
    int n_slots = 0;
    uint8_t* q_raw = nullptr;               // n_slots * nb * 64 bytes (64B aligned)
    float* dy_raw = nullptr;                // n_slots * nb * 16 floats (64B aligned)
    size_t cap_blocks = 0;

    Nvfp4PackedActs512() = default;
    ~Nvfp4PackedActs512();
    Nvfp4PackedActs512(const Nvfp4PackedActs512&) = delete;
    Nvfp4PackedActs512& operator=(const Nvfp4PackedActs512&) = delete;

    void prepare(int slots, int n_elems);
};

// Quantize FP32 row `x` (length `n`, multiple of 64) directly into packed slot `slot` of `dst`.
void nvfp4_quantize_and_pack_f32_avx512(const float* x, int slot, int n, Nvfp4PackedActs512& dst);

// Pack existing Q8_0 row into packed slot `slot` of `dst`.
void nvfp4_pack_q8_0_one_avx512(const uint8_t* q8_row, int slot, int n, Nvfp4PackedActs512& dst);

// Fused gate + up + activation for rows [r0, r1) (multiples of 64 when fused_pack_dst != nullptr)
// of ONE expert used by `nt` tokens.
// - `tok_rows[0..nt-1]`: input activation slot indices in `in_acts`
// - `exp_slots[0..nt-1]`: output expert indices (for `fused_pack_dst` and/or `h_out`)
// When `fused_pack_dst != nullptr`, quantizes the 64 activated values directly into
// `fused_pack_dst` at `(slot = exp_slots[t], ib = r0 / 64)` without touching `h_out`.
void nvfp4_gu_rows_avx512(const uint8_t* gate, const uint8_t* up, size_t row_bytes, int n_embd,
                          const Nvfp4PackedActs512& in_acts, const int* tok_rows, const int* exp_slots, int nt,
                          float gate_scale, float up_scale, float limit, bool gelu, SiTU situ,
                          Nvfp4PackedActs512* fused_pack_dst, float* h_out, int n_ff, int r0, int r1);

// Fused multi-expert NVFP4 down projection for output rows [r0, r1) of `n_embd`.
// Accumulates router-weighted expert outputs into `out[row * n_embd + r]` in exact expert order.
void nvfp4_down_rows_avx512(const CpuExpert* experts, const int* by, const int* run, int n_run,
                            int n_experts, int rows, size_t d_row, int n_ff, int n_embd,
                            const Nvfp4PackedActs512& h_acts, float* out, int r0, int r1);

}  // namespace strata::poly
