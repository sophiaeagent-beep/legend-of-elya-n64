/*
 * kv_compress_test.c — N64 on-device smoke test for kv_compress.h
 *
 * Prints: RMSE of roundtrip Q8 and Q4 over a random K-like vector,
 * relative error on a dot product, and the memory footprint delta.
 *
 * Build with:
 *   $(CC) $(CFLAGS) -o build/kv_compress_test kv_compress_test.c -lm
 * Then run under ares / EverDrive, or compile host-side with -DHOST for
 * a faster CI-style roundtrip check.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kv_compress.h"

#ifndef HOST
#include <libdragon.h>
#endif

#define VDIM 256   /* must match KVC_DIM */

/* Deterministic synthetic K vector: mixture of signs + magnitudes that
 * approximates a typical post-projection K distribution. */
static void fill_synthetic_k(float *v, unsigned seed) {
    unsigned s = seed ? seed : 0xC001D00Du;
    for (int i = 0; i < VDIM; i++) {
        s = s * 1664525u + 1013904223u;
        float u = (float)(s & 0xFFFFFF) / (float)0xFFFFFF;  /* [0,1) */
        u = u * 2.0f - 1.0f;                                 /* [-1,1) */
        v[i] = u * 1.7f;                                     /* rough K scale */
    }
}

static void print_report(const char *label, float rmse, float max_abs, float dot_err_pct) {
#ifdef HOST
    printf("%-6s  RMSE=%.5f  max|v|=%.5f  dot_err=%.3f%%\n",
           label, rmse, max_abs, dot_err_pct);
#else
    /* N64: debugf via isviewer/USB */
    debugf("%s  RMSE=%f  max|v|=%f  dot_err=%f%%\n",
           label, rmse, max_abs, dot_err_pct);
#endif
}

int main(void) {
#ifndef HOST
    console_init();
    console_set_render_mode(RENDER_MANUAL);
    debug_init_isviewer();
    debug_init_usblog();
#endif

    static float k_orig[VDIM];
    static float k_back[VDIM];
    static float q_vec [VDIM];
    static KVCQ4Vec k4;
    static KVCQ8Vec k8;

    fill_synthetic_k(k_orig, 0x12345678u);
    fill_synthetic_k(q_vec,  0x87654321u);  /* independent "query" vector */

    float vmax = kvc_vec_max_abs(k_orig, VDIM);

    /* ---- Q8 roundtrip ---- */
    kvc_q8_pack(k_orig, &k8);
    kvc_q8_unpack(&k8, k_back);
    float q8_rmse = kvc_vec_rmse(k_orig, k_back, VDIM);

    /* Reference dot product (float32) */
    float ref_dot = 0.0f;
    for (int i = 0; i < VDIM; i++) ref_dot += q_vec[i] * k_orig[i];

    float q8_dot = kvc_q8_dot(q_vec, &k8, VDIM);
    float q8_dot_err = 100.0f * ((q8_dot - ref_dot) / (ref_dot == 0 ? 1e-8f : ref_dot));
    if (q8_dot_err < 0) q8_dot_err = -q8_dot_err;

    print_report("Q8", q8_rmse, vmax, q8_dot_err);

    /* ---- Q4 roundtrip ---- */
    kvc_q4_pack(k_orig, &k4);
    kvc_q4_unpack(&k4, k_back);
    float q4_rmse = kvc_vec_rmse(k_orig, k_back, VDIM);

    float q4_dot = kvc_q4_dot(q_vec, &k4, VDIM);
    float q4_dot_err = 100.0f * ((q4_dot - ref_dot) / (ref_dot == 0 ? 1e-8f : ref_dot));
    if (q4_dot_err < 0) q4_dot_err = -q4_dot_err;

    print_report("Q4", q4_rmse, vmax, q4_dot_err);

    /* ---- Footprint math (v9 LARGE) ---- */
    long bytes_fp32 = 8L * 128 * 256 * 4 * 2;              /* 2 MiB */
    long bytes_q8   = 8L * 128 * (256 + 4) * 2;            /* int8 + fp32 scale per vec */
    long bytes_q4   = 8L * 128 * (128 + 4) * 2;            /* 4-bit packed + fp32 scale */

#ifdef HOST
    printf("\nKV cache footprint (v9 LARGE: 8L x CTX128 x DIM256):\n");
    printf("  fp32 : %ld bytes (%ld KB)\n", bytes_fp32, bytes_fp32 / 1024);
    printf("  Q8   : %ld bytes (%ld KB)  -> %.2fx reclaim\n",
           bytes_q8, bytes_q8 / 1024, (float)bytes_fp32 / (float)bytes_q8);
    printf("  Q4   : %ld bytes (%ld KB)  -> %.2fx reclaim\n",
           bytes_q4, bytes_q4 / 1024, (float)bytes_fp32 / (float)bytes_q4);
#else
    debugf("KV footprint fp32=%ld KB  Q8=%ld KB  Q4=%ld KB\n",
           bytes_fp32 / 1024, bytes_q8 / 1024, bytes_q4 / 1024);
    /* Keep screen up so numbers are visible on real hardware */
    while (1) { /* halt */ }
#endif

    return 0;
}
