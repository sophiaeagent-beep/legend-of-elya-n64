/*
 * libdragon_host.h — minimal stub of libdragon.h for host (x86-64 Linux) builds
 *
 * Satisfies #include <libdragon.h> in nano_gpt.c without pulling in any N64 code.
 * -include this file when compiling nano_gpt.c on a host machine.
 */
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

/* n64sys.h provides this on N64 — no-op on host */
static inline void data_cache_hit_writeback_invalidate(volatile void *addr,
                                                        unsigned long size)
{
    (void)addr; (void)size;
}

/* memalign: POSIX provides this in <stdlib.h> with _XOPEN_SOURCE 700 */
/* nano_gpt.c uses it once; our gen_sophia_host.c provides static storage instead */
