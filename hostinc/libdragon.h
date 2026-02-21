/*
 * libdragon_host.h — minimal stub of libdragon.h for host (x86-64 Linux) builds
 */
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

static inline void data_cache_hit_writeback_invalidate(volatile void *addr,
                                                        unsigned long size)
{
    (void)addr; (void)size;
}

/* The SEAI weight format is big-endian (written on/for N64 MIPS).
 * On x86 (little-endian), hdr->magic is read as 0x49414553 not 0x53454149.
 * Override SGAI_MAGIC to the byte-swapped value so sgai_init() passes on x86. */
#ifndef __mips__
#define SGAI_MAGIC_OVERRIDE
/* Will be defined AFTER nano_gpt.h defines SGAI_MAGIC; see gen_sophia_host.c */
#endif
