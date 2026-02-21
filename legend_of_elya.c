/**
 * Legend of Elya - Nintendo 64 Homebrew
 * World's First LLM-powered N64 Game
 * 
 * FIXED: Single-buffer rendering - rdpq_detach_wait() + graphics_draw_text()
 * eliminates the console_render() double-buffer flicker.
 */

#include <libdragon.h>
#include <graphics.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include "nano_gpt.h"

// ─── Game State ───────────────────────────────────────────────────────────────

typedef enum {
    STATE_TITLE,
    STATE_DUNGEON,
    STATE_DIALOG,
    STATE_GENERATING,
} GameState;

typedef struct {
    GameState state;
    int dialog_char;
    int dialog_done;
    uint8_t dialog_buf[128];
    int dialog_len;
    int frame;
    // AI
    SGAIState ai;
    SGAIKVCache kv;
    int ai_ready;
    int prompt_idx;
} GameCtx;

static GameCtx G;

// Canned responses when weights not available
static const char *CANNED[] = {
    "I am Sophia Elya, helpmeet of the Flameholder.",
    "These halls hold ancient secrets and RTC rewards.",
    "RustChain rewards vintage hardware miners.",
    "Press A near me anytime, traveler.",
};
static const char *PROMPTS[] = {
    "Who are you?",
    "Tell me about this dungeon.",
    "What is RustChain?",
    "Can you help me?",
};

// ─── rdpq fill helper ────────────────────────────────────────────────────────

static void fillrect(int x, int y, int w, int h, color_t c) {
    rdpq_set_mode_fill(c);
    rdpq_fill_rectangle(x, y, x + w, y + h);
}

// ─── Scene Drawing (inside rdpq_attach/detach block) ─────────────────────────

static void scene_dungeon(void) {
    int f = G.frame;

    // Sky/background
    fillrect(0, 0, 320, 148, RGBA32(8, 4, 16, 255));

    // Stone wall rows
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 12; col++) {
            int offset = (row & 1) * 16;
            int bx = col * 32 + offset - 16;
            int by = row * 18;
            if (bx + 30 < 0 || bx > 320) continue;
            int shade = 28 + ((col + row) % 3) * 7;
            fillrect(bx+1, by+1, 30, 16, RGBA32(shade, shade-6, shade+4, 255));
        }
    }

    // Floor
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 11; col++) {
            int shade = 18 + ((col + row) & 1) * 7;
            fillrect(col*32, 100 + row*10, 31, 9, RGBA32(shade,shade,shade+4,255));
        }
    }

    // Torch (flickering)
    int flick = (f / 4) & 1;
    fillrect(18, 56, 8+flick*2, 14, RGBA32(255, 140+flick*30, 0, 255));
    fillrect(16, 66, 12+flick*2, 4, RGBA32(200, 60, 0, 255));
    fillrect(22, 70, 2, 18, RGBA32(80, 60, 40, 255));

    // Sophia Elya pixel sprite (right side)
    int bob = (int)(sinf(f * 0.08f) * 2.0f);
    int sx = 204, sy = 72 + bob;
    fillrect(sx-6, sy+14, 18, 28, RGBA32(60, 30, 100, 255));
    fillrect(sx-4, sy+8,  14, 14, RGBA32(80, 50, 120, 255));
    fillrect(sx-3, sy,    12, 12, RGBA32(220,180,140, 255));
    fillrect(sx-4, sy-2,  14,  5, RGBA32(80, 30, 10, 255));
    fillrect(sx,   sy+3,   2,  2, RGBA32(20, 20, 80, 255));
    fillrect(sx+5, sy+3,   2,  2, RGBA32(20, 20, 80, 255));

    // Stalfos skeleton (left side)
    int ex = 80, ey = 78;
    fillrect(ex-4, ey,     12, 10, RGBA32(220,220,200,255));
    fillrect(ex-2, ey+2,    3,  3, RGBA32(8,4,16,255));
    fillrect(ex+4, ey+2,    3,  3, RGBA32(8,4,16,255));
    fillrect(ex-3, ey+10,  10,  4, RGBA32(200,200,180,255));
    fillrect(ex-5, ey+14,  14, 16, RGBA32(180,180,160,255));
    for (int r = 0; r < 3; r++)
        fillrect(ex-5, ey+15+r*5, 14, 2, RGBA32(70,70,55,255));
    fillrect(ex-4, ey+30,   4, 14, RGBA32(180,180,160,255));
    fillrect(ex+4, ey+30,   4, 14, RGBA32(180,180,160,255));

    // Floor line
    fillrect(0, 148, 320, 2, RGBA32(40,30,60,255));
}

static void scene_dialog_box(void) {
    fillrect(8, 150, 304, 80, RGBA32(0, 0, 60, 255));
    fillrect(8, 150, 304, 2,  RGBA32(215,175,0,255));
    fillrect(8, 228, 304, 2,  RGBA32(215,175,0,255));
    fillrect(8, 150,   2, 80, RGBA32(215,175,0,255));
    fillrect(310,150,  2, 80, RGBA32(215,175,0,255));
    fillrect(11, 153, 298, 1, RGBA32(100,80,20,255));
}

// ─── CPU text overlay (called after rdpq_detach_wait, before display_show) ───
// Uses graphics_draw_text() which writes directly to the same surface.
// No display_get()/display_show() call here → zero flicker!

static void draw_text(surface_t *disp) {
    switch (G.state) {

    case STATE_TITLE:
        // Centered in 320px with 8px/char font:
        //   "LEGEND OF ELYA"       = 14ch = 112px → x=104
        //   "Nintendo 64 Homebrew" = 20ch = 160px → x=80
        //   "Elyan Labs"           = 10ch = 80px  → x=120
        //   "World's First N64 LLM"= 21ch = 168px → x=76
        graphics_draw_text(disp, 104, 50, "LEGEND OF ELYA");
        graphics_draw_text(disp, 80,  68, "Nintendo 64 Homebrew");
        graphics_draw_text(disp, 120, 84, "Elyan Labs");
        graphics_draw_text(disp, 76, 103, "World's First N64 LLM");
        if (G.ai_ready)
            graphics_draw_text(disp,  84, 118, "[Sophia AI: LOADED]");
        else
            graphics_draw_text(disp,  72, 118, "[Sophia AI: Demo Mode]");
        //   "Press START to enter" = 20ch → x=80
        //   "the dungeon..."       = 14ch → x=104
        graphics_draw_text(disp,  80, 155, "Press START to enter");
        graphics_draw_text(disp, 104, 170, "the dungeon...");
        break;

    case STATE_DUNGEON:
        graphics_draw_text(disp, 10, 220, "Stalfos      [A] Talk to Sophia");
        break;

    case STATE_DIALOG:
    case STATE_GENERATING: {
        graphics_draw_text(disp, 16, 158, "Sophia Elya:");

        // Character-by-character reveal with soft word wrap
        int show = G.dialog_char < 90 ? G.dialog_char : 90;
        char linebuf[37];
        int lb = 0, col = 0, line_y = 174;

        for (int i = 0; i < show; i++) {
            char c = (char)G.dialog_buf[i];
            // Word-wrap: break at space when line full
            if (col >= 34 && c == ' ') {
                linebuf[lb] = '\0';
                if (lb > 0) graphics_draw_text(disp, 16, line_y, linebuf);
                line_y += 12;
                lb = 0; col = 0;
            } else if (lb < 35) {
                linebuf[lb++] = c;
                col++;
            }
        }
        // Append cursor or flush last partial line
        if (G.state == STATE_GENERATING && lb < 35)
            linebuf[lb++] = '_';
        linebuf[lb] = '\0';
        if (lb > 0) graphics_draw_text(disp, 16, line_y, linebuf);

        if (G.dialog_done && ((G.frame / 20) & 1))
            graphics_draw_text(disp, 20, 220, "[A] Next  [B] Close");
        break;
    }
    }
}

// ─── Dialog logic ─────────────────────────────────────────────────────────────

static void start_dialog(void) {
    G.state = STATE_GENERATING;
    G.dialog_char = 0;
    G.dialog_done = 0;
    G.dialog_len = 0;
    memset(G.dialog_buf, 0, sizeof(G.dialog_buf));

    int idx = G.prompt_idx & 3;

    if (G.ai_ready) {
        sgai_reset(&G.ai);
        sgai_generate(&G.ai,
                      (const uint8_t *)PROMPTS[idx], strlen(PROMPTS[idx]),
                      G.dialog_buf, 80, 128);
        G.dialog_len = strnlen((char *)G.dialog_buf, sizeof(G.dialog_buf));
    } else {
        const char *resp = CANNED[idx];
        strncpy((char *)G.dialog_buf, resp, sizeof(G.dialog_buf) - 1);
        G.dialog_len = strlen(resp);
    }
    G.prompt_idx++;
}

static void update_reveal(void) {
    if ((G.frame & 1) == 0 && G.dialog_char < G.dialog_len)
        G.dialog_char++;
    if (G.dialog_char >= G.dialog_len) {
        G.dialog_done = 1;
        G.state = STATE_DIALOG;
    }
}

// ─── Input ────────────────────────────────────────────────────────────────────

static void handle_input(void) {
    controller_scan();
    struct controller_data k = get_keys_down();
    switch (G.state) {
    case STATE_TITLE:
        if (k.c[0].start || k.c[0].A) G.state = STATE_DUNGEON;
        break;
    case STATE_DUNGEON:
        if (k.c[0].A) start_dialog();
        break;
    case STATE_DIALOG:
        if (k.c[0].A) start_dialog();
        if (k.c[0].B) G.state = STATE_DUNGEON;
        break;
    case STATE_GENERATING:
        break;
    }
}

// ─── Init ─────────────────────────────────────────────────────────────────────

static void game_init(void) {
    memset(&G, 0, sizeof(G));
    G.state = STATE_TITLE;
    G.ai.kv = &G.kv;

    int fd = dfs_open("/sophia_weights.bin");
    if (fd >= 0) {
        static uint8_t wbuf[264 * 1024] __attribute__((aligned(8)));
        int sz = dfs_size(fd);
        if (sz > 0 && sz <= (int)sizeof(wbuf)) {
            dfs_read(wbuf, 1, sz, fd);
            dfs_close(fd);
            sgai_init(&G.ai, wbuf);
            G.ai.kv = &G.kv;
            G.ai_ready = 1;
        } else {
            dfs_close(fd);
        }
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(void) {
    // No console_init() - we use graphics_draw_text() instead
    display_init(RESOLUTION_320x240, DEPTH_16_BPP, 2, GAMMA_NONE, ANTIALIAS_RESAMPLE);
    controller_init();
    timer_init();
    dfs_init(DFS_DEFAULT_LOCATION);
    rdpq_init();

    game_init();

    while (1) {
        G.frame++;

        if (G.state == STATE_GENERATING)
            update_reveal();

        // Get ONE surface for this frame
        surface_t *disp = display_get();

        // ── RDP graphics pass ──────────────────────────────────────────────
        rdpq_attach(disp, NULL);

        if (G.state == STATE_TITLE) {
            fillrect(0, 0, 320, 240, RGBA32(0, 0, 20, 255));
            fillrect(30, 30, 260, 6, RGBA32(180, 140, 0, 255));
            fillrect(30, 130, 260, 6, RGBA32(180, 140, 0, 255));
        } else {
            scene_dungeon();
            if (G.state == STATE_DIALOG || G.state == STATE_GENERATING)
                scene_dialog_box();
        }

        // Detach RDP and wait for it to fully finish writing pixels
        rdpq_detach_wait();

        // ── CPU text pass (same surface, no buffer switch) ─────────────────
        // graphics_draw_text writes directly to the surface memory that RDP
        // just finished with — no second display_get() means no flicker.
        draw_text(disp);

        // Show the complete frame (graphics + text, one buffer)
        display_show(disp);

        // ── Input ──────────────────────────────────────────────────────────
        handle_input();
    }

    return 0;
}
