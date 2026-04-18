#pragma once
/*
 * kv_compress.h — TurboQuant-style KV cache compression helpers
 *
 * Status: Phase 1 drop-in module. Pack/unpack primitives only.
 *         Not yet wired into nano_gpt.c — see kv_compress_test.c for
 *         on-device roundtrip accuracy verification.
 *
 * Motivation
 * ----------
 * v9 LARGE uses 8 layers × CTX 128 × 256 embed × 4 bytes × 2 (K+V) = 2 MB
 * KV cache. Quantizing each vector to 4 bits + per-vector fp32 scale cuts
 * this to ~260 KB — roughly 8x reclaim, unlocking CTX growth or deeper
 * models without leaving Expansion Pak territory.
 *
 * Design
 * ------
 * Per-vector symmetric Q4:
 *   scale = max(abs(v)) / 7.0          (range = -7..+7 as signed 4-bit)
 *   q[i]  = round(v[i] / scale)        (clipped to [-7,7], then +7 to [0,14])
 *   dequant: v[i] = (q[i] - 7) * scale
 *
 * Packing: two 4-bit values per byte, low nibble first.
 *
 * Q8 variant is also provided for an intermediate 4x compression option
 * that keeps per-coordinate fidelity higher, useful as a fallback if 4-bit
 * accuracy is too lossy in the attention dot product.
 *
 * Walsh-Hadamard pre-rotation is deferred to Phase 2: it whitens the
 * coordinate distribution, making per-element scalar quantization more
 * robust. On R4300i it's effectively free (sign flips + adds, no FPU mul).
 */

#include <stdint.h>
#include <math.h>

/* Vector dimension matches SGAI_N_EMBED in nano_gpt.h */
#ifndef KVC_DIM
#define KVC_DIM 256
#endif

/* Compile-time constants ---------------------------------------------- */
#define KVC_Q4_PACKED_BYTES (KVC_DIM / 2)
#define KVC_Q8_PACKED_BYTES (KVC_DIM)

/* Q4 symmetric: [-7, +7] shifted to [0, 14] for unsigned storage */
typedef struct {
    uint8_t q[KVC_Q4_PACKED_BYTES];  /* 2 nibbles per byte */
    float   scale;                   /* per-vector dequant multiplier */
} KVCQ4Vec;

/* Q8 symmetric: int8 [-127, +127] (we avoid -128 to keep |q|=scale*127) */
typedef struct {
    int8_t q[KVC_Q8_PACKED_BYTES];
    float  scale;
} KVCQ8Vec;

/* ---------------- Q4 pack/unpack ---------------- */

static inline void kvc_q4_pack(const float *src, KVCQ4Vec *dst) {
    float amax = 0.0f;
    for (int i = 0; i < KVC_DIM; i++) {
        float a = src[i] < 0 ? -src[i] : src[i];
        if (a > amax) amax = a;
    }
    float scale = amax / 7.0f;
    if (scale <= 0.0f) scale = 1e-8f;  /* avoid div-by-zero on all-zero vector */
    float inv = 1.0f / scale;
    dst->scale = scale;

    for (int i = 0; i < KVC_DIM; i += 2) {
        /* round-to-nearest, clip to [-7, +7], shift to [0, 14] */
        int q0 = (int)(src[i    ] * inv + (src[i    ] >= 0 ? 0.5f : -0.5f));
        int q1 = (int)(src[i + 1] * inv + (src[i + 1] >= 0 ? 0.5f : -0.5f));
        if (q0 >  7) q0 =  7; else if (q0 < -7) q0 = -7;
        if (q1 >  7) q1 =  7; else if (q1 < -7) q1 = -7;
        uint8_t n0 = (uint8_t)(q0 + 7) & 0x0F;
        uint8_t n1 = (uint8_t)(q1 + 7) & 0x0F;
        dst->q[i >> 1] = (uint8_t)((n1 << 4) | n0);
    }
}

static inline void kvc_q4_unpack(const KVCQ4Vec *src, float *dst) {
    float s = src->scale;
    for (int i = 0; i < KVC_DIM; i += 2) {
        uint8_t b  = src->q[i >> 1];
        int     q0 = (int)(b & 0x0F) - 7;
        int     q1 = (int)(b >> 4)   - 7;
        dst[i    ] = (float)q0 * s;
        dst[i + 1] = (float)q1 * s;
    }
}

/* ---------------- Q8 pack/unpack ---------------- */

static inline void kvc_q8_pack(const float *src, KVCQ8Vec *dst) {
    float amax = 0.0f;
    for (int i = 0; i < KVC_DIM; i++) {
        float a = src[i] < 0 ? -src[i] : src[i];
        if (a > amax) amax = a;
    }
    float scale = amax / 127.0f;
    if (scale <= 0.0f) scale = 1e-8f;
    float inv = 1.0f / scale;
    dst->scale = scale;

    for (int i = 0; i < KVC_DIM; i++) {
        int q = (int)(src[i] * inv + (src[i] >= 0 ? 0.5f : -0.5f));
        if (q >  127) q =  127; else if (q < -127) q = -127;
        dst->q[i] = (int8_t)q;
    }
}

static inline void kvc_q8_unpack(const KVCQ8Vec *src, float *dst) {
    float s = src->scale;
    for (int i = 0; i < KVC_DIM; i++) dst[i] = (float)src->q[i] * s;
}

/* ---------------- Dot-product helpers (fused) ----------------
 *
 * The attention inner loop needs:
 *     sum_{d} q[d] * k[d]
 *
 * With packed K we fuse dequant into the dot. This avoids materialising
 * a scratch float[DIM] buffer during the head-wise reduction.
 */

static inline float kvc_q4_dot(const float *qvec, const KVCQ4Vec *kvec, int dim) {
    float s   = kvec->scale;
    float acc = 0.0f;
    for (int i = 0; i < dim; i += 2) {
        uint8_t b  = kvec->q[i >> 1];
        int     k0 = (int)(b & 0x0F) - 7;
        int     k1 = (int)(b >> 4)   - 7;
        acc += qvec[i    ] * (float)k0;
        acc += qvec[i + 1] * (float)k1;
    }
    return acc * s;
}

static inline float kvc_q8_dot(const float *qvec, const KVCQ8Vec *kvec, int dim) {
    float s   = kvec->scale;
    float acc = 0.0f;
    for (int i = 0; i < dim; i++) acc += qvec[i] * (float)kvec->q[i];
    return acc * s;
}

/* ---------------- Roundtrip error helpers (for testing) ---------------- */

static inline float kvc_vec_rmse(const float *a, const float *b, int dim) {
    float sse = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sse += d * d;
    }
    return (float)sqrt((double)(sse / (double)dim));
}

static inline float kvc_vec_max_abs(const float *a, int dim) {
    float m = 0.0f;
    for (int i = 0; i < dim; i++) {
        float x = a[i] < 0 ? -a[i] : a[i];
        if (x > m) m = x;
    }
    return m;
}
