/**
 * Legend of Elya - Nintendo 64 Homebrew
 * World's First LLM-powered N64 Game
 *
 * FIXED: Single-buffer rendering - rdpq_detach_wait() + graphics_draw_text()
 * eliminates the console_render() double-buffer flicker.
 *
 * v2: Legend of Elya splash screen with balloons + per-token tok/s indicator
 * v3: LOZ Dungeon Theme square-wave music via libdragon audio
 */

#include <libdragon.h>
#include <graphics.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include "nano_gpt.h"

#ifdef USE_RPC_LLM
#include "n64_llm_rpc.h"
#endif

// ─── Game State ───────────────────────────────────────────────────────────────

typedef enum {
    STATE_ANNIVERSARY,   // Legend of Elya splash screen
    STATE_TITLE,
    STATE_DUNGEON,
    STATE_DIALOG,
    STATE_GENERATING,
    STATE_KEYBOARD,      // Virtual keyboard for player text input
    STATE_GAMEOVER,      // Death / retry screen
    STATE_TRANSITION,    // Between-room blackout fade
    STATE_VICTORY,       // Boss slain
} GameState;

// ─── Gameplay: movement, enemies, pickups ──────────────────────────────────
// Playfield is 320x240. HUD occupies y=0..13, floor line at y=149.
// Walkable band for Sophia: y in [SOPHIA_Y_MIN .. SOPHIA_Y_MAX].
#define SOPHIA_Y_MIN    60
#define SOPHIA_Y_MAX    132
#define SOPHIA_X_MIN    16
#define SOPHIA_X_MAX    296
#define SOPHIA_SPEED    2      // pixels per frame
#define SOPHIA_IFRAMES  48     // ~0.8s invulnerability after hit
#define SWING_FRAMES    20     // total attack anim length
#define SWING_HIT_START 5      // hit window frame start
#define SWING_HIT_END   14     // hit window frame end
#define ATTACK_RANGE    32     // pixel distance A-press targets enemy

typedef enum {
    FACE_DOWN  = 0,
    FACE_UP    = 1,
    FACE_LEFT  = 2,
    FACE_RIGHT = 3,
} Facing;

typedef enum {
    ENEMY_NONE    = 0,
    ENEMY_STALFOS = 1,
    ENEMY_KEESE   = 2,
    ENEMY_BOSS    = 3,   // Big Stalfos — HP 8, telegraphed lunge
} EnemyKind;

typedef struct {
    int kind;        // EnemyKind; NONE = slot free
    int x, y;        // center position
    int hp;          // 0 = dead
    int max_hp;
    int hurt_timer;  // frames of hit-flash remaining
    int ai_timer;    // kind-specific AI cooldown
    int vx, vy;      // velocity (keese flight)
    int spawn_x, spawn_y;  // home position (stalfos pacing)
} Enemy;

typedef enum {
    PICKUP_NONE   = 0,
    PICKUP_RUPEE  = 1,   // green, +1
    PICKUP_RUPEE5 = 2,   // blue,  +5
    PICKUP_HEART  = 3,   // +2 half-hearts
} PickupKind;

typedef struct {
    int kind;
    int x, y;
    int vx, vy;      // pop-out velocity (settles to 0 on floor)
    int life;        // frames remaining before despawn
} Pickup;

#define MAX_ENEMIES 5
#define MAX_PICKUPS 8

// ─── Rooms ─────────────────────────────────────────────────────────────────
#define N_ROOMS              3
#define ROOM_SPAWNS_MAX      5
#define TRANSITION_FRAMES    28   // fade-out + swap + fade-in

typedef struct {
    int kind;
    int x, y;
} RoomSpawn;

typedef struct {
    int n_spawns;
    RoomSpawn spawns[ROOM_SPAWNS_MAX];
    /* Decor colour tweaks per room (BG shade tint) */
    uint8_t bg_r, bg_g, bg_b;
    int torch_x;     // left torch x position
} RoomDef;

static const RoomDef ROOMS[N_ROOMS] = {
    /* Room 0 — Entrance Hall */
    {
        .n_spawns = 2,
        .spawns = {
            { ENEMY_STALFOS,  80,  95 },
            { ENEMY_KEESE,   140,  55 },
        },
        .bg_r = 8, .bg_g = 4, .bg_b = 16,
        .torch_x = 18,
    },
    /* Room 1 — The Crypt */
    {
        .n_spawns = 4,
        .spawns = {
            { ENEMY_STALFOS,  50, 115 },
            { ENEMY_STALFOS, 240,  95 },
            { ENEMY_KEESE,   110,  50 },
            { ENEMY_KEESE,   200,  62 },
        },
        .bg_r = 12, .bg_g = 4, .bg_b = 8,
        .torch_x = 290,
    },
    /* Room 2 — Boss Arena */
    {
        .n_spawns = 1,
        .spawns = {
            { ENEMY_BOSS, 160, 105 },
        },
        .bg_r = 14, .bg_g = 4, .bg_b = 22,
        .torch_x = 18,
    },
};

typedef struct {
    GameState state;
    int dialog_char;
    int dialog_done;
    uint8_t dialog_buf[128];
    int dialog_len;
    int frame;
    uint32_t anniversary_cp0;  // CP0 Count at boot for real-time splash duration
    // AI
    SGAIState ai;
    SGAIKVCache kv;
    int ai_ready;
    int prompt_idx;
    // Per-frame generation state (enables tok/s display)
    uint8_t gen_pbuf[64];   // copy of current prompt bytes
    int gen_plen;           // prompt byte count
    int gen_ppos;           // bytes fed so far (prompt phase)
    uint8_t gen_last_tok;   // last token for chaining
    int gen_out_count;      // output tokens generated
    int gen_start_frame;    // frame when output phase began
    float gen_toks_sec;     // computed tokens/second
    // Music sequencer
    int music_note_idx;     // current note in sequence
    int music_sample_pos;   // samples elapsed in current note
    int music_phase;        // square wave phase accumulator
    // Combat & HUD
    int attack_timer;       // frames remaining in attack animation (0 = idle)
    int attack_target;      // 0 = stalfos, 1 = keese
    int hearts;             // half-heart count: 8 = 4 full hearts, 0 = dead
    int magic;              // magic bar 0-128 (128 = full)
    // Performance monitoring
    uint32_t perf_frame_start;   // CP0 COUNT at frame start
    uint32_t perf_gen_cycles;    // cycles spent in sgai_next_token this frame
    uint32_t perf_gen_total_us;  // total generation time in microseconds
    uint32_t perf_gen_start_us;  // timestamp when generation started
    float    perf_cpu_pct;       // CPU% used by inference (0-100)
    float    perf_toks_precise;  // precise tok/s using cycle counter
    int      perf_show;          // 1 = show performance overlay
    // Virtual keyboard
    int      kb_row;             // cursor row (0-3)
    int      kb_col;             // cursor column (0-9)
    char     kb_input[64];       // player input buffer
    int      kb_len;             // current input length
    int      kb_debounce;        // frame counter for input debounce
#ifdef USE_RPC_LLM
    int      rpc_active;         // 1 = bridge detected, using RPC
    int      rpc_pending;        // 1 = waiting for RPC response
    uint32_t rpc_send_us;        // timestamp when RPC request sent
#endif
    // ── Gameplay ─────────────────────────────────────────────────────────
    int      sophia_x, sophia_y;   // player sprite center (was hardcoded 204,72)
    int      sophia_face;          // Facing enum
    int      sophia_iframes;       // invincibility frames remaining (hit cooldown)
    int      swing_timer;          // 0 = idle, >0 = active sword swing frames remaining
    int      swing_hit_done;       // 1 = already registered hit this swing
    int      rupees;               // collected currency
    int      kills;                // total enemies slain (stats)
    Enemy    enemies[MAX_ENEMIES];
    Pickup   pickups[MAX_PICKUPS];
    int      game_over_frame;      // frame when hearts hit 0 (for fade-in timing)
    // ── Rooms ────────────────────────────────────────────────────────────
    int      room;                 // current room index 0..N_ROOMS-1
    int      room_cleared[N_ROOMS];// 1 = all enemies defeated in that room
    int      transition_frame;     // counts up during STATE_TRANSITION
    int      transition_dir;       // +1 = going east, -1 = going west
    int      transition_target;    // destination room index
    int      boss_hp_max;          // cached max HP for boss bar rendering
    int      victory_frame;        // frame boss was slain (for fanfare fade)
} GameCtx;

static GameCtx G;

// ─── Gameplay helpers ────────────────────────────────────────────────────────

static inline int iabs(int a) { return a < 0 ? -a : a; }

static int find_enemy_in_range(int x, int y, int range) {
    int best = -1;
    int best_d2 = range * range;
    for (int i = 0; i < MAX_ENEMIES; i++) {
        if (G.enemies[i].kind == ENEMY_NONE || G.enemies[i].hp <= 0) continue;
        int dx = G.enemies[i].x - x;
        int dy = G.enemies[i].y - y;
        int d2 = dx*dx + dy*dy;
        if (d2 < best_d2) { best_d2 = d2; best = i; }
    }
    return best;
}

static void spawn_pickup(int x, int y, int kind) {
    for (int i = 0; i < MAX_PICKUPS; i++) {
        if (G.pickups[i].kind == PICKUP_NONE) {
            G.pickups[i].kind = kind;
            G.pickups[i].x = x;
            G.pickups[i].y = y;
            /* Use CP0 cycle counter low bits for pop direction (no sinf cost) */
            uint32_t e = (uint32_t)TICKS_READ() ^ (uint32_t)(i * 0x9E37u);
            G.pickups[i].vx = ((int)(e & 0x7) - 3);       // -3..+4
            G.pickups[i].vy = -4 - (int)((e >> 4) & 0x3); // -4..-7 (upward)
            G.pickups[i].life = 300;                      // 5 seconds at 60fps
            return;
        }
    }
}

static void spawn_enemy(int slot, int kind, int x, int y) {
    G.enemies[slot].kind       = kind;
    G.enemies[slot].x          = x;
    G.enemies[slot].y          = y;
    G.enemies[slot].spawn_x    = x;
    G.enemies[slot].spawn_y    = y;
    G.enemies[slot].hurt_timer = 0;
    G.enemies[slot].ai_timer   = 0;
    G.enemies[slot].vx         = 0;
    G.enemies[slot].vy         = 0;
    switch (kind) {
        case ENEMY_STALFOS: G.enemies[slot].hp = G.enemies[slot].max_hp = 3; break;
        case ENEMY_KEESE:   G.enemies[slot].hp = G.enemies[slot].max_hp = 1; break;
        case ENEMY_BOSS:    G.enemies[slot].hp = G.enemies[slot].max_hp = 8; break;
        default:            G.enemies[slot].hp = G.enemies[slot].max_hp = 0; break;
    }
}

/* Load room idx: populate enemies[] from RoomDef (unless already cleared),
 * clear pickups, cache boss HP for bar rendering. */
static void load_room(int idx) {
    if (idx < 0) idx = 0;
    if (idx >= N_ROOMS) idx = N_ROOMS - 1;
    G.room = idx;
    memset(G.enemies, 0, sizeof(G.enemies));
    memset(G.pickups, 0, sizeof(G.pickups));
    G.boss_hp_max = 0;
    if (G.room_cleared[idx]) return;  /* revisit: stay empty */
    const RoomDef *rd = &ROOMS[idx];
    int n = rd->n_spawns; if (n > MAX_ENEMIES) n = MAX_ENEMIES;
    for (int i = 0; i < n; i++) {
        spawn_enemy(i, rd->spawns[i].kind, rd->spawns[i].x, rd->spawns[i].y);
        if (rd->spawns[i].kind == ENEMY_BOSS) G.boss_hp_max = G.enemies[i].max_hp;
    }
}

static void reset_dungeon(void) {
    /* Full run reset — called on boot and retry. Start in room 0. */
    memset(G.room_cleared, 0, sizeof(G.room_cleared));
    G.sophia_x = 240;
    G.sophia_y = 110;
    G.sophia_face = FACE_LEFT;
    G.sophia_iframes = 0;
    G.swing_timer = 0;
    G.swing_hit_done = 0;
    G.hearts = 8;
    G.magic  = 128;
    G.rupees = 0;
    G.kills  = 0;
    G.game_over_frame = 0;
    G.transition_frame = 0;
    G.transition_dir = 0;
    G.transition_target = 0;
    G.victory_frame = 0;
    load_room(0);
}

/* Count live enemies in current room (used to mark room_cleared) */
static int live_enemy_count(void) {
    int n = 0;
    for (int i = 0; i < MAX_ENEMIES; i++)
        if (G.enemies[i].kind != ENEMY_NONE && G.enemies[i].hp > 0) n++;
    return n;
}

static void damage_player(int half_hearts) {
    if (G.sophia_iframes > 0) return;
    G.hearts -= half_hearts;
    if (G.hearts < 0) G.hearts = 0;
    G.sophia_iframes = SOPHIA_IFRAMES;
    if (G.hearts == 0) {
        G.state = STATE_GAMEOVER;
        G.game_over_frame = G.frame;
    }
}

static void hit_enemy(int idx) {
    Enemy *e = &G.enemies[idx];
    if (e->hp <= 0) return;
    e->hp--;
    e->hurt_timer = 12;
    if (e->hp <= 0) {
        /* Death drops: vary by enemy kind */
        uint32_t e_rng = (uint32_t)TICKS_READ() ^ (uint32_t)(idx * 0x45D9F3B1u);
        int was_boss = (e->kind == ENEMY_BOSS);
        int drop = PICKUP_RUPEE;
        if (e->kind == ENEMY_STALFOS && (e_rng & 0x3) == 0)  drop = PICKUP_RUPEE5;
        else if (e->kind == ENEMY_STALFOS && (e_rng & 0xF) == 1) drop = PICKUP_HEART;
        else if (was_boss) drop = PICKUP_RUPEE5;  /* primary boss drop */
        spawn_pickup(e->x, e->y, drop);
        if (was_boss) {
            /* Boss bonus: extra rupee5 volley + heart refill */
            spawn_pickup(e->x + 8, e->y - 4, PICKUP_RUPEE5);
            spawn_pickup(e->x - 8, e->y - 4, PICKUP_RUPEE5);
            spawn_pickup(e->x,     e->y - 10, PICKUP_HEART);
            G.victory_frame = G.frame;
            /* Transition to victory after brief delay handled in update_world */
        }
        G.kills++;
        e->kind = ENEMY_NONE;  /* free slot */
        /* Room-clear check: if no live enemies remain, lock it as cleared */
        if (live_enemy_count() == 0) G.room_cleared[G.room] = 1;
    }
}

static void update_enemies(void) {
    for (int i = 0; i < MAX_ENEMIES; i++) {
        Enemy *e = &G.enemies[i];
        if (e->kind == ENEMY_NONE || e->hp <= 0) continue;
        if (e->hurt_timer > 0) e->hurt_timer--;
        e->ai_timer++;

        if (e->kind == ENEMY_STALFOS) {
            /* Patrol toward Sophia when within 80px; else pace home. */
            int dx = G.sophia_x - e->x;
            int dy = G.sophia_y - e->y;
            int d2 = dx*dx + dy*dy;
            if (d2 < 80*80 && (e->ai_timer & 1)) {
                if (dx >  1) e->x += 1;
                else if (dx < -1) e->x -= 1;
                if (dy >  1) e->y += 1;
                else if (dy < -1) e->y -= 1;
            } else if ((e->ai_timer / 30) & 1) {
                /* Pace 12px either side of spawn */
                if (e->x < e->spawn_x + 12) e->x += (e->ai_timer & 1);
                else if (e->x > e->spawn_x - 12) e->x -= (e->ai_timer & 1);
            }
            /* Clamp to floor band */
            if (e->y < SOPHIA_Y_MIN) e->y = SOPHIA_Y_MIN;
            if (e->y > SOPHIA_Y_MAX) e->y = SOPHIA_Y_MAX;
        } else if (e->kind == ENEMY_KEESE) {
            /* Periodic dive-bomb toward Sophia (every ~90 frames),
             * otherwise return to drift height. */
            if ((e->ai_timer % 90) < 30) {
                int dx = G.sophia_x - e->x;
                int dy = G.sophia_y - e->y;
                if (dx >  1) e->x += 2;
                else if (dx < -1) e->x -= 2;
                if (dy >  1) e->y += 1;
                else if (dy < -1) e->y -= 1;
            } else {
                /* Orbit drift — sine via cheap lookup */
                e->x = e->spawn_x + (int)(sinf(e->ai_timer * 0.05f) * 40.0f);
                e->y = 55 + (int)(sinf(e->ai_timer * 0.08f) * 14.0f);
            }
        } else if (e->kind == ENEMY_BOSS) {
            /* Boss pattern: 90-frame cycle.
             *   0-40:  windup — steps slowly toward Sophia, raises arms
             *   41-60: LUNGE — fast dash toward player (telegraph already done)
             *   61-90: recover — drift back toward spawn, vulnerable window */
            int phase = e->ai_timer % 90;
            int dx = G.sophia_x - e->x;
            int dy = G.sophia_y - e->y;
            if (phase < 40) {
                /* Slow windup approach */
                if ((phase & 3) == 0) {
                    if (dx >  1) e->x += 1;
                    else if (dx < -1) e->x -= 1;
                    if (dy >  1) e->y += 1;
                    else if (dy < -1) e->y -= 1;
                }
            } else if (phase < 60) {
                /* LUNGE — 3px per frame toward player */
                if (dx >  1) e->x += 3;
                else if (dx < -1) e->x -= 3;
                if (dy >  1) e->y += 2;
                else if (dy < -1) e->y -= 2;
            } else {
                /* Recover — drift back to spawn */
                if (e->x < e->spawn_x) e->x += 1;
                else if (e->x > e->spawn_x) e->x -= 1;
                if (e->y < e->spawn_y) e->y += 1;
                else if (e->y > e->spawn_y) e->y -= 1;
            }
            if (e->x < 20)  e->x = 20;
            if (e->x > 300) e->x = 300;
            if (e->y < SOPHIA_Y_MIN) e->y = SOPHIA_Y_MIN;
            if (e->y > SOPHIA_Y_MAX) e->y = SOPHIA_Y_MAX;
        }

        /* Contact damage: Sophia overlaps enemy box */
        int cdx = G.sophia_x - e->x;
        int cdy = G.sophia_y - e->y;
        int reach = (e->kind == ENEMY_BOSS) ? 26 : 18;
        if (iabs(cdx) < reach && iabs(cdy) < reach) {
            /* Boss hits for full heart, minions for half */
            damage_player((e->kind == ENEMY_BOSS) ? 2 : 1);
        }
    }
}

static void update_pickups(void) {
    for (int i = 0; i < MAX_PICKUPS; i++) {
        Pickup *p = &G.pickups[i];
        if (p->kind == PICKUP_NONE) continue;
        /* Ballistic pop with floor gravity */
        p->x  += p->vx;
        p->y  += p->vy;
        p->vy += 1;  /* gravity */
        if (p->y > 140) { p->y = 140; p->vy = 0; p->vx = 0; }  /* rest on floor */
        p->life--;
        if (p->life <= 0) { p->kind = PICKUP_NONE; continue; }
        /* Collection */
        int dx = p->x - G.sophia_x;
        int dy = p->y - G.sophia_y;
        if (iabs(dx) < 14 && iabs(dy) < 18) {
            switch (p->kind) {
                case PICKUP_RUPEE:  G.rupees += 1;  break;
                case PICKUP_RUPEE5: G.rupees += 5;  break;
                case PICKUP_HEART:
                    G.hearts += 2;
                    if (G.hearts > 8) G.hearts = 8;
                    break;
            }
            p->kind = PICKUP_NONE;
        }
    }
}

/* Resolve a sword-swing hit window once per swing. Called each frame while
 * G.swing_timer > 0. Uses facing direction to place the hitbox in front of
 * Sophia, and only triggers a hit once per swing (swing_hit_done latch). */
static void resolve_swing_hit(void) {
    if (G.swing_timer <= 0) return;
    int prog = SWING_FRAMES - G.swing_timer;  /* 0 at start, grows */
    if (prog < SWING_HIT_START || prog > SWING_HIT_END) return;
    if (G.swing_hit_done) return;

    int hx = G.sophia_x, hy = G.sophia_y;
    switch (G.sophia_face) {
        case FACE_UP:    hy -= 22; break;
        case FACE_DOWN:  hy += 22; break;
        case FACE_LEFT:  hx -= 22; break;
        case FACE_RIGHT: hx += 22; break;
    }
    int idx = find_enemy_in_range(hx, hy, 22);
    if (idx >= 0) {
        hit_enemy(idx);
        G.swing_hit_done = 1;
    }
}

/* Begin a room transition: blackout fade → swap room → fade in.
 * dir is +1 (east, next room) or -1 (west, previous room). */
static void begin_transition(int dir) {
    if (dir > 0 && G.room >= N_ROOMS - 1) return;  /* at east end */
    if (dir < 0 && G.room <= 0)           return;  /* at west end */
    /* Block progression east unless current room cleared */
    if (dir > 0 && !G.room_cleared[G.room]) return;
    G.state            = STATE_TRANSITION;
    G.transition_frame = 0;
    G.transition_dir   = dir;
    G.transition_target= G.room + dir;
}

static void update_world(void) {
    /* Victory: if boss room is cleared, latch into victory state (after brief
     * pause so drops can be collected visually) */
    if (G.state == STATE_DUNGEON && G.room_cleared[N_ROOMS - 1]
        && G.victory_frame > 0 && (G.frame - G.victory_frame) > 120) {
        G.state = STATE_VICTORY;
    }

    if (G.state == STATE_TRANSITION) {
        G.transition_frame++;
        /* At midpoint, swap rooms and teleport Sophia to opposite edge */
        if (G.transition_frame == TRANSITION_FRAMES / 2) {
            load_room(G.transition_target);
            if (G.transition_dir > 0) {
                G.sophia_x = SOPHIA_X_MIN + 6;   /* enter from west side */
                G.sophia_face = FACE_RIGHT;
            } else {
                G.sophia_x = SOPHIA_X_MAX - 6;   /* enter from east side */
                G.sophia_face = FACE_LEFT;
            }
            /* Centre vertically on re-entry */
            G.sophia_y = 110;
            G.sophia_iframes = 30;  /* brief spawn invincibility */
        }
        if (G.transition_frame >= TRANSITION_FRAMES) {
            G.state = STATE_DUNGEON;
            G.transition_frame = 0;
        }
        return;
    }

    if (G.state != STATE_DUNGEON) return;

    if (G.sophia_iframes > 0) G.sophia_iframes--;
    if (G.swing_timer > 0) {
        resolve_swing_hit();
        G.swing_timer--;
        if (G.swing_timer == 0) G.swing_hit_done = 0;
    }
    update_enemies();
    update_pickups();

    /* Magic regen (slow): 1 unit every 8 frames */
    if ((G.frame & 7) == 0 && G.magic < 128) G.magic++;

    /* Door trigger: walk off east/west edge with conditions met */
    if (G.sophia_x <= SOPHIA_X_MIN + 2 && G.room > 0) {
        begin_transition(-1);
    } else if (G.sophia_x >= SOPHIA_X_MAX - 2
               && G.room < N_ROOMS - 1
               && G.room_cleared[G.room]) {
        begin_transition(+1);
    }
}

/* N64 hardware entropy — XOR CPU cycle counter low bits with frame,
 * last token, AND prompt_idx (sequential counter).
 * prompt_idx is the critical fallback: emulators run TICKS_READ()
 * deterministically, so we rely on prompt_idx advancing each A-press
 * to guarantee a different topic every conversation.
 * On real hardware, TICKS jitter adds extra unpredictability. */
/* N64 R4300i runs at 93.75 MHz. CP0 COUNT increments every other cycle = 46.875 MHz.
 * So 1 microsecond = ~46.875 counts.  We use TICKS_READ() which reads CP0 COUNT. */
#define CYCLES_TO_US(c)   ((uint32_t)(c) / 47)
#define US_TO_CYCLES(us)  ((uint32_t)(us) * 47)

/* One NTSC frame = 1/60s = 16667us = ~781,250 CP0 counts */
#define FRAME_CYCLES  781250

#define N64_ENTROPY() ((uint32_t)(TICKS_READ())                    \
                       ^ ((uint32_t)G.frame << 3)                  \
                       ^ ((uint32_t)G.gen_last_tok * 2654435761u)  \
                       ^ ((uint32_t)G.prompt_idx  * 40503u))

/* Post-generation output filter — remove training data artifacts.
 * "helpmeet" was in the QA training corpus but is wrong for a game.
 * In-place replacement keeps same buffer length (no shift needed).
 * "guardian" is exactly 8 chars = same as "helpmeet". */
static void filter_dialog_buf(void) {
    char *buf = (char *)G.dialog_buf;
    /* replace "helpmeet" → "guardian" (8 chars = 8 chars, in-place safe) */
    char *p = buf;
    while ((p = strstr(p, "helpmeet")) != NULL) {
        memcpy(p, "guardian", 8);
        p += 8;
    }
    /* replace "Flameholder" → "Elyan Labs " (11 chars = 11 chars) */
    p = buf;
    while ((p = strstr(p, "Flameholder")) != NULL) {
        memcpy(p, "Elyan Labs ", 11);
        p += 11;
    }
}

// Fallback responses when weights not available (no helpmeet/title language)
static const char *CANNED[] = {
    "I am Sophia Elya, guide of the realm.",
    "Vintage hardware earns real RTC rewards.",
    "The G4 and G5 are my favorite miners.",
    "RustChain proves old silicon still matters.",
    "The VR4300 inside this cartridge is real.",
    "Seek the silver key behind the great statue.",
    "Many adventurers have braved these halls.",
    "Elyan Labs built me to run on 8 megabytes.",
    "The RSP and RDP team up to draw these halls.",
    "PowerPC G4 earns two point five times RTC.",
    "Three attestation nodes guard the network.",
    "Ancient silicon dreams in proof of antiquity.",
    "The legend of Elya endures in silicon.",
    "This dungeon holds secrets only brave find.",
    "I was trained on 50 thousand steps of lore.",
    "Press A near me anytime, weary traveler.",
};
#define N_CANNED 16

/* Prompt pool — exact QA_PAIRS keys from training data.
 * v5 CTX=64 gives room for prompts up to ~20 chars,
 * leaving 44+ tokens for Sophia's response.
 * Entropy from N64 CPU oscillator selects the prompt each conversation. */
static const char *PROMPTS[] = {
    /* identity */
    "Who are you?: ",
    "What is your name?: ",
    "Where are you from?: ",
    "What is your purpose?: ",
    /* dungeon / game */
    "What lurks here?: ",
    "How do I proceed?: ",
    "What do I need here?: ",
    "Tell me a secret.: ",
    /* RustChain */
    "What is RustChain?: ",
    "What is RTC?: ",
    "How do I earn RTC?: ",
    "What is a node?: ",
    "What is proof of antiquity?: ",
    "What is epoch?: ",
    /* hardware */
    "What is the G4?: ",
    "What is the G5?: ",
    "What is POWER8?: ",
    "What is AltiVec?: ",
    "What is vec_perm?: ",
    "What runs this ROM?: ",
    "What is the VR4300?: ",
    /* N64 lore */
    "What console is this?: ",
    "What is MIPS?: ",
    "How big is your model?: ",
    "What language runs you?: ",
    /* Elya lore */
    "What is Elyan Labs?: ",
    "Who is the Helpmeet?: ",
    "What is the Study?: ",
    "Who guards the realm?: ",
    "What is the Triforce?: ",
    "Who is the Flameholder?: ",
    "What is proof of work?: ",
};
#define N_PROMPTS 32

// ─── Music: Legend of Elya Theme (Original) ──────────────────────────────────
// Original composition for Legend of Elya.
// Key: A minor / C major modal. BPM ~110. Mysterious dungeon atmosphere
// with a wistful, exploratory feel. Rising minor arpeggio opening,
// chromatic tension, then resolution through the natural minor scale.
// Notes: A4=440, C5=523, D5=587, E5=659, F5=698, G5=784, A5=880,
//        B4=494, Bb4=466, G4=392, F4=349, E4=330
// 0 = rest/silence

#define MUSIC_FREQ         22050        // 22kHz, plenty for square wave
#define MUSIC_BPM          110
#define MUSIC_EIGHTH       (MUSIC_FREQ * 60 / (MUSIC_BPM * 2))   // ~6013 samples
#define MUSIC_ATTACK       350          // samples of fade-in per note
#define MUSIC_DECAY_START  (MUSIC_EIGHTH - 450) // start fade-out near end

static const uint16_t DUNGEON_FREQ[] = {
    // Phrase 1: mysterious ascending A minor arpeggio
    330,  440,  523,  659,   // E4 A4 C5 E5  (Am arpeggio, wistful)
    523,  659,  523,    0,   // C5 E5 C5 rest
    // Phrase 2: tension — chromatic climb F5→G5, then descend
    698,  784,  880,    0,   // F5 G5 A5 rest  (climax, high A)
    784,  698,    0,  587,   // G5 F5 rest D5   (falling back)
    // Phrase 3: melancholy descent through natural minor
    659,  587,  523,    0,   // E5 D5 C5 rest  (stepwise descent)
    494,  523,  587,  659,   // B4 C5 D5 E5   (re-ascend, hope)
    // Phrase 4: resolution — settle into tonic with gentle fade
    880,    0,  659,    0,   // A5 rest E5 rest  (octave call)
    523,  440,    0,    0,   // C5 A4 rest rest  (home)
};
#define DUNGEON_LEN 32

// Duration: 1=eighth note, 2=quarter note (held)
static const uint8_t DUNGEON_DUR[] = {
    // Phrase 1
    1, 1, 1, 2,   // E4 A4 C5 E5(held)  (lingering on the 5th)
    1, 2, 1, 1,   // C5 E5(held) C5 rest
    // Phrase 2
    1, 1, 2, 1,   // F5 G5 A5(held) rest  (sustained climax)
    1, 2, 1, 1,   // G5 F5(held) rest D5
    // Phrase 3
    1, 1, 2, 1,   // E5 D5 C5(held) rest
    1, 1, 1, 1,   // B4 C5 D5 E5  (quick run back up)
    // Phrase 4
    2, 1, 2, 1,   // A5(held) rest E5(held) rest
    2, 2, 1, 1,   // C5(held) A4(held) rest rest
};

static void music_update(void) {
    if (!audio_can_write()) return;

    short *buf = audio_write_begin();
    int nsamples = audio_get_buffer_length();

    for (int i = 0; i < nsamples; i++) {
        int note_samples = (int)DUNGEON_DUR[G.music_note_idx] * MUSIC_EIGHTH;
        uint16_t freq    = DUNGEON_FREQ[G.music_note_idx];

        int16_t sample = 0;
        if (freq > 0) {
            int period = MUSIC_FREQ / (int)freq;
            if (period > 0) {
                // Square wave
                int16_t amp = 5000;
                // Simple attack/decay envelope to avoid clicks
                if (G.music_sample_pos < MUSIC_ATTACK)
                    amp = (int16_t)((int32_t)amp * G.music_sample_pos / MUSIC_ATTACK);
                else if (G.music_sample_pos > MUSIC_DECAY_START)
                    amp = (int16_t)((int32_t)amp * (note_samples - G.music_sample_pos)
                                    / (note_samples - MUSIC_DECAY_START));
                sample = (G.music_phase < period / 2) ? amp : -amp;
                G.music_phase = (G.music_phase + 1) % period;
            }
        } else {
            G.music_phase = 0;
        }

        buf[i * 2]     = sample;   // left
        buf[i * 2 + 1] = sample;   // right

        // Advance note timer
        if (++G.music_sample_pos >= note_samples) {
            G.music_sample_pos = 0;
            G.music_phase      = 0;
            G.music_note_idx   = (G.music_note_idx + 1) % DUNGEON_LEN;
        }
    }

    audio_write_end();
}

// ─── rdpq fill helper ────────────────────────────────────────────────────────

static void fillrect(int x, int y, int w, int h, color_t c) {
    rdpq_set_mode_fill(c);
    rdpq_fill_rectangle(x, y, x + w, y + h);
}

// ─── Balloon drawing (in RDP pass) ───────────────────────────────────────────

// Festive balloon colors
static const color_t BALLOON_COLORS[6] = {
    {255, 60,  60,  255},   // red
    {255, 160, 30,  255},   // orange
    {240, 220, 0,   255},   // yellow
    {60,  210, 80,  255},   // green
    {60,  140, 255, 255},   // blue
    {220, 60,  255, 255},   // purple
};

// Curved string offsets (precomputed, avoids sinf per pixel)
static const int STRING_DX[14] = { 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, -1, 0, 0 };

static void draw_balloon(int cx, int cy, color_t c) {
    // Oval body using layered horizontal rects
    fillrect(cx-4,  cy-10,  9,  2, c);
    fillrect(cx-7,  cy-8,  15,  2, c);
    fillrect(cx-9,  cy-6,  19,  3, c);
    fillrect(cx-10, cy-3,  21,  3, c);   // widest
    fillrect(cx-10, cy,    21,  3, c);   // widest
    fillrect(cx-9,  cy+3,  19,  3, c);
    fillrect(cx-7,  cy+6,  15,  2, c);
    fillrect(cx-4,  cy+8,   9,  2, c);
    // Knot
    fillrect(cx-2,  cy+10,  5,  3, c);
    // Curvy string (precomputed offsets, no sinf)
    for (int i = 0; i < 14; i++)
        fillrect(cx + STRING_DX[i], cy+13+i, 1, 1, RGBA32(190, 190, 190, 255));
}

// ─── Anniversary Scene (RDP pass) ─────────────────────────────────────────────

// Balloon x positions and frame-phase offsets for variety
static const int BALLOON_X[6]     = { 28, 72, 118, 165, 210, 262 };
static const int BALLOON_PHASE[6] = {  0, 40,  80,  20,  60,  10 };

static void scene_anniversary(void) {
    int f = G.frame;

    // Deep blue-black starry background
    fillrect(0, 0, 320, 240, RGBA32(4, 4, 28, 255));

    // Twinkling stars (deterministic positions, brightness flickers)
    for (int i = 0; i < 32; i++) {
        int sx = (i * 97 + 13) % 316 + 2;
        int sy = (i * 53 + 7)  % 195 + 2;
        int bright = 80 + (((f + i * 17) >> 3) & 1) * 120;
        fillrect(sx, sy, 1, 1, RGBA32(bright, bright, bright, 255));
    }

    // Elya crystal gem (cyan/teal, centered around x=152)
    {
        int cx = 152, cy = 28;
        color_t gem_bright = RGBA32(80, 220, 255, 255);
        color_t gem_mid    = RGBA32(40, 160, 200, 255);
        color_t gem_dark   = RGBA32(20, 100, 160, 255);
        // Top facet — narrow peak widening to center
        for (int row = 0; row < 12; row++) {
            int w = row * 2 + 2;
            color_t c = (row < 4) ? gem_bright : gem_mid;
            fillrect(cx - row, cy + row*2, w, 2, c);
        }
        // Bottom facet — widest at center, narrowing to point
        for (int row = 0; row < 14; row++) {
            int w = 24 - row * 2;
            if (w < 2) w = 2;
            color_t c = (row > 10) ? gem_bright : gem_dark;
            fillrect(cx - w/2, cy + 24 + row*2, w, 2, c);
        }
    }

    // Floating balloons - each drifts upward at slightly different speed
    for (int i = 0; i < 6; i++) {
        int period = 200 + i * 20;   // frames to cross full height
        int raw_y  = 270 - (((f + BALLOON_PHASE[i]) % period) * 270 / period);
        // Gentle horizontal sway using frame counter (no sinf)
        int sway = ((f + BALLOON_PHASE[i]) >> 3) & 1 ? 2 : -2;
        if (raw_y > -30 && raw_y < 245) {
            draw_balloon(BALLOON_X[i] + sway, raw_y, BALLOON_COLORS[i]);
        }
    }

    // Gold border
    fillrect(0,   0,   320, 3, RGBA32(215, 175, 0, 255));
    fillrect(0,   237, 320, 3, RGBA32(215, 175, 0, 255));
    fillrect(0,   0,   3, 240, RGBA32(215, 175, 0, 255));
    fillrect(317, 0,   3, 240, RGBA32(215, 175, 0, 255));
}

// ─── Dungeon Scene ────────────────────────────────────────────────────────────

static void draw_stalfos(int ex, int ey, int flash) {
    fillrect(ex-4, ey,     12, 10, RGBA32(220,220,200,255));
    fillrect(ex-2, ey+2,    3,  3, RGBA32(8,4,16,255));
    fillrect(ex+4, ey+2,    3,  3, RGBA32(8,4,16,255));
    fillrect(ex-3, ey+10,  10,  4, RGBA32(200,200,180,255));
    fillrect(ex-5, ey+14,  14, 16, RGBA32(180,180,160,255));
    for (int r = 0; r < 3; r++)
        fillrect(ex-5, ey+15+r*5, 14, 2, RGBA32(70,70,55,255));
    fillrect(ex-4, ey+30,   4, 14, RGBA32(180,180,160,255));
    fillrect(ex+4, ey+30,   4, 14, RGBA32(180,180,160,255));
    if (flash) fillrect(ex-5, ey, 14, 44, RGBA32(255, 255, 255, 200));
}

/* Big Stalfos boss: 2× Stalfos sprite, crown, red eyes. Windup pose = arms
 * raised (phase<40 of 90-frame cycle), lunge/recovery = normal pose. */
static void draw_boss(int ex, int ey, int ai_timer, int flash) {
    int phase = ai_timer % 90;
    int armup = (phase < 40);
    color_t bone_hi = RGBA32(235,235,215,255);
    color_t bone    = RGBA32(205,205,185,255);
    color_t bone_lo = RGBA32(170,170,150,255);
    color_t skull_eye = RGBA32(255, 30,  30, 255);
    color_t crown   = RGBA32(215,175,  0, 255);
    /* Skull (bigger than regular stalfos) */
    fillrect(ex-8,  ey-4,    20, 14, bone_hi);
    fillrect(ex-6,  ey,       4,  4, skull_eye);
    fillrect(ex+2,  ey,       4,  4, skull_eye);
    fillrect(ex-4,  ey+6,     2,  3, RGBA32(8,4,16,255));
    fillrect(ex,    ey+6,     2,  3, RGBA32(8,4,16,255));
    fillrect(ex+4,  ey+6,     2,  3, RGBA32(8,4,16,255));
    /* Crown */
    fillrect(ex-8,  ey-8,     4,  4, crown);
    fillrect(ex-2,  ey-10,    4,  6, crown);
    fillrect(ex+4,  ey-8,     4,  4, crown);
    /* Neck + ribcage */
    fillrect(ex-5,  ey+10,   10,  3, bone);
    fillrect(ex-10, ey+13,   20, 24, bone);
    for (int r = 0; r < 5; r++)
        fillrect(ex-10, ey+14+r*5, 20, 2, bone_lo);
    /* Arms: up when winding up, out to sides otherwise */
    if (armup) {
        fillrect(ex-14, ey-2,     4, 22, bone);   /* left arm raised */
        fillrect(ex+10, ey-2,     4, 22, bone);   /* right arm raised */
    } else {
        fillrect(ex-16, ey+14,    6,  4, bone);   /* left arm out */
        fillrect(ex+10, ey+14,    6,  4, bone);   /* right arm out */
        fillrect(ex-20, ey+18,    4, 10, bone_lo);/* hand */
        fillrect(ex+16, ey+18,    4, 10, bone_lo);/* hand */
    }
    /* Legs (wider stance) */
    fillrect(ex-8,  ey+37,    6, 16, bone);
    fillrect(ex+2,  ey+37,    6, 16, bone);
    fillrect(ex-9,  ey+52,    7,  3, bone_lo);
    fillrect(ex+1,  ey+52,    7,  3, bone_lo);
    if (flash) fillrect(ex-16, ey-10, 32, 68, RGBA32(255, 255, 255, 200));
}

static void draw_keese(int kx, int ky, int wing, int flash) {
    /* Dark body */
    fillrect(kx-3, ky-2, 6, 5, RGBA32(25, 15, 35, 255));
    if (wing == 0) {
        fillrect(kx-12, ky-5,  9, 6, RGBA32(50, 35, 70, 255));
        fillrect(kx+3,  ky-5,  9, 6, RGBA32(50, 35, 70, 255));
        fillrect(kx-12, ky-1,  4, 3, RGBA32(35, 22, 50, 255));
        fillrect(kx+8,  ky-1,  4, 3, RGBA32(35, 22, 50, 255));
    } else {
        fillrect(kx-12, ky+1,  9, 5, RGBA32(50, 35, 70, 255));
        fillrect(kx+3,  ky+1,  9, 5, RGBA32(50, 35, 70, 255));
        fillrect(kx-10, ky-2,  4, 3, RGBA32(35, 22, 50, 255));
        fillrect(kx+6,  ky-2,  4, 3, RGBA32(35, 22, 50, 255));
    }
    fillrect(kx-1, ky-1, 2, 2, RGBA32(255, 60,  20, 255));
    fillrect(kx+2, ky-1, 2, 2, RGBA32(255, 60,  20, 255));
    if (flash) fillrect(kx-12, ky-5, 25, 13, RGBA32(255, 255, 255, 200));
}

static void draw_pickup(const Pickup *p, int f) {
    int twinkle = (f / 6) & 1;
    switch (p->kind) {
        case PICKUP_RUPEE: {  /* green rhombus */
            color_t base = RGBA32(60, 200, 110, 255);
            color_t hi   = RGBA32(180, 255, 200, 255);
            fillrect(p->x-1, p->y-4, 2, 2, base);
            fillrect(p->x-2, p->y-2, 4, 2, base);
            fillrect(p->x-3, p->y,   6, 2, base);
            fillrect(p->x-2, p->y+2, 4, 2, base);
            fillrect(p->x-1, p->y+4, 2, 2, base);
            if (twinkle) fillrect(p->x,   p->y-2, 1, 2, hi);
            break;
        }
        case PICKUP_RUPEE5: {  /* blue, slightly bigger */
            color_t base = RGBA32(60, 140, 255, 255);
            color_t hi   = RGBA32(180, 220, 255, 255);
            fillrect(p->x-1, p->y-5, 3, 2, base);
            fillrect(p->x-3, p->y-3, 7, 2, base);
            fillrect(p->x-4, p->y-1, 9, 2, base);
            fillrect(p->x-3, p->y+1, 7, 2, base);
            fillrect(p->x-1, p->y+3, 3, 2, base);
            if (twinkle) fillrect(p->x,   p->y-3, 1, 3, hi);
            break;
        }
        case PICKUP_HEART: {   /* red heart pickup */
            color_t red = RGBA32(230,  50,  50, 255);
            color_t hi  = RGBA32(255, 180, 180, 255);
            fillrect(p->x-3, p->y-3, 2, 2, red);
            fillrect(p->x+1, p->y-3, 2, 2, red);
            fillrect(p->x-4, p->y-1, 8, 3, red);
            fillrect(p->x-3, p->y+2, 6, 1, red);
            fillrect(p->x-2, p->y+3, 4, 1, red);
            fillrect(p->x-1, p->y+4, 2, 1, red);
            if (twinkle) fillrect(p->x-2, p->y-2, 1, 1, hi);
            break;
        }
    }
}

static void scene_dungeon(void) {
    int f = G.frame;

    /* Legacy hit-flash countdown (still drives slash/spark overlays) */
    int atk = G.attack_timer;
    if (atk > 0) G.attack_timer--;

    /* Full-screen clear — N64 double-buffers, so leaving any pixel unpainted
     * shows stale content from the previous frame's buffer. Paint every
     * pixel below the HUD. */
    fillrect(0, 150, 320, 90, RGBA32(6, 3, 12, 255));

    // Sky/background — tinted per room
    const RoomDef *rd = &ROOMS[G.room];
    fillrect(0, 0, 320, 148, RGBA32(rd->bg_r, rd->bg_g, rd->bg_b, 255));

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

    // Torch (flickering) — position per room
    int flick = (f / 4) & 1;
    int tx0 = rd->torch_x;
    fillrect(tx0,     56, 8+flick*2, 14, RGBA32(255, 140+flick*30, 0, 255));
    fillrect(tx0-2,   66, 12+flick*2, 4, RGBA32(200, 60, 0, 255));
    fillrect(tx0+4,   70, 2, 18, RGBA32(80, 60, 40, 255));

    // Sophia Elya pixel sprite — now driven by G.sophia_x/y + facing
    int bob = (int)(sinf(f * 0.08f) * 2.0f);
    int sx = G.sophia_x, sy = G.sophia_y + bob;
    /* i-frame flicker: hide sprite every other frame while invincible */
    int hide_sprite = (G.sophia_iframes > 0) && ((f >> 1) & 1);
    /* Shield/sword side follow facing: LEFT flips both to right-hand side */
    int face_left = (G.sophia_face == FACE_LEFT);
    int sh_off = face_left ? +13 : -20;   /* shield offset from body center */
    int sw_off = face_left ? -15 :  13;   /* sword offset */

    if (!hide_sprite) {
    // ── Shield (left arm, kite-style) ────────────────────────────────────
    {
        int shx = sx + sh_off, shy = sy + 6;
        // Kite-shaped body (wider in middle, narrows to point)
        fillrect(shx+1, shy,      10,  2, RGBA32(20,  50, 130, 255));
        fillrect(shx,   shy+2,    12, 10, RGBA32(20,  50, 130, 255));
        fillrect(shx+1, shy+12,   10,  4, RGBA32(20,  50, 130, 255));
        fillrect(shx+2, shy+16,    8,  3, RGBA32(20,  50, 130, 255));
        fillrect(shx+4, shy+19,    4,  4, RGBA32(20,  50, 130, 255));
        // Gold trim border
        fillrect(shx,   shy,      12,  2, RGBA32(215,175,  0, 255));
        fillrect(shx,   shy+2,     2, 20, RGBA32(215,175,  0, 255));
        fillrect(shx+10,shy+2,     2, 20, RGBA32(215,175,  0, 255));
        // Elya gem emblem (mini crystal)
        fillrect(shx+5, shy+4,     2,  2, RGBA32(80,220,255, 255));  // top
        fillrect(shx+3, shy+6,     6,  3, RGBA32(40,160,200, 255));  // middle
        fillrect(shx+5, shy+9,     2,  2, RGBA32(80,220,255, 255));  // bottom
    }

    // ── Sophia body ───────────────────────────────────────────────────────
    fillrect(sx-6, sy+14, 18, 28, RGBA32(60, 30, 100, 255));  // dress
    fillrect(sx-4, sy+8,  14, 14, RGBA32(80, 50, 120, 255));  // torso
    fillrect(sx-3, sy,    12, 12, RGBA32(220,180,140, 255));   // head
    fillrect(sx-4, sy-2,  14,  5, RGBA32(80, 30, 10, 255));   // hair
    fillrect(sx,   sy+3,   2,  2, RGBA32(20, 20, 80, 255));   // left eye
    fillrect(sx+5, sy+3,   2,  2, RGBA32(20, 20, 80, 255));   // right eye

    // ── Sword (follows facing; swings during G.swing_timer) ──────────────
    {
        int swx = sx + sw_off;
        /* Swing tilt: during swing, blade rotates from raised → forward.
         * Approximate by shifting blade_y down as swing progresses. */
        int tilt = 0;
        if (G.swing_timer > 0) {
            int prog = SWING_FRAMES - G.swing_timer;  /* 0..19 */
            tilt = (prog < 10) ? prog * 2 : (20 - prog) * 2;  /* 0→20→0 */
        }
        int blade_top_y = sy - 16 + tilt;
        // Blade (silver)
        fillrect(swx,   blade_top_y,     2, 28, RGBA32(195,215,235, 255));
        fillrect(swx+1, blade_top_y,     1, 14, RGBA32(240,250,255, 255));
        // Blade tip
        fillrect(swx,   blade_top_y - 2, 2,  2, RGBA32(220,235,250, 255));
        // Crossguard (gold) — fixed at handle level
        fillrect(swx-5, sy+12,  12,  3, RGBA32(215,175,  0, 255));
        fillrect(swx-5, sy+12,  12,  1, RGBA32(255,220, 60, 255));
        // Handle
        fillrect(swx,   sy+15,   2,  7, RGBA32(110, 60, 15, 255));
        // Pommel (gold ball)
        fillrect(swx-1, sy+22,   4,  3, RGBA32(215,175,  0, 255));
        // Animated gleam — only when idle (not during swing, too busy)
        if (G.swing_timer == 0) {
            int gleam = (f / 7) % 22;
            fillrect(swx, sy + 10 - gleam, 1, 4, RGBA32(255,255,255, 200));
        }
    }
    }  /* end if (!hide_sprite) */

    // ── Enemies (dynamic, via G.enemies[]) ──────────────────────────────────
    int wing = (f / 5) & 1;
    for (int i = 0; i < MAX_ENEMIES; i++) {
        const Enemy *e = &G.enemies[i];
        if (e->kind == ENEMY_NONE || e->hp <= 0) continue;
        int flash = (e->hurt_timer > 0);
        if (e->kind == ENEMY_STALFOS)     draw_stalfos(e->x, e->y - 22, flash);
        else if (e->kind == ENEMY_KEESE)  draw_keese(e->x, e->y, wing, flash);
        else if (e->kind == ENEMY_BOSS)   draw_boss(e->x, e->y - 28, e->ai_timer, flash);
    }

    /* Boss HP bar — shows in room 2 while boss alive */
    if (G.room == N_ROOMS - 1 && G.boss_hp_max > 0) {
        int boss_idx = -1;
        for (int i = 0; i < MAX_ENEMIES; i++) {
            if (G.enemies[i].kind == ENEMY_BOSS && G.enemies[i].hp > 0) {
                boss_idx = i; break;
            }
        }
        if (boss_idx >= 0) {
            int hp = G.enemies[boss_idx].hp;
            int bar_w = 200;
            int bar_x = (320 - bar_w) / 2;
            int bar_y = 20;
            int fill_w = (bar_w * hp) / G.boss_hp_max;
            fillrect(bar_x - 2, bar_y - 2, bar_w + 4, 10, RGBA32(30, 10, 10, 255));
            fillrect(bar_x, bar_y, bar_w, 6, RGBA32(60, 10, 10, 255));
            if (fill_w > 0)
                fillrect(bar_x, bar_y, fill_w, 6, RGBA32(220, 40, 40, 255));
            /* Gold border */
            fillrect(bar_x - 2, bar_y - 2, bar_w + 4, 1, RGBA32(215,175,0,255));
            fillrect(bar_x - 2, bar_y + 7, bar_w + 4, 1, RGBA32(215,175,0,255));
        }
    }

    // ── Pickups (rupees, hearts) ────────────────────────────────────────────
    for (int i = 0; i < MAX_PICKUPS; i++) {
        if (G.pickups[i].kind != PICKUP_NONE) draw_pickup(&G.pickups[i], f);
    }

    // ── Treasure Chest ─────────────────────────────────────────────────────
    // Sits on the floor, center-right of dungeon
    {
        int cx = 146, cy = 112;
        // Chest body (lower)
        fillrect(cx,    cy+10, 28, 20, RGBA32(100, 62, 18, 255));
        // Chest lid (upper, slightly lighter)
        fillrect(cx,    cy,    28, 12, RGBA32(130, 82, 28, 255));
        // Curved lid top highlight
        fillrect(cx+2,  cy-1,  24,  2, RGBA32(155, 100, 40, 255));
        // Gold trim — horizontal bands
        fillrect(cx,    cy+10, 28,  2, RGBA32(215, 175,  0, 255));  // lid seam
        fillrect(cx,    cy+28, 28,  2, RGBA32(215, 175,  0, 255));  // bottom
        fillrect(cx,    cy,    28,  2, RGBA32(215, 175,  0, 255));  // top
        // Gold trim — vertical sides
        fillrect(cx,    cy,     2, 30, RGBA32(215, 175,  0, 255));  // left
        fillrect(cx+26, cy,     2, 30, RGBA32(215, 175,  0, 255));  // right
        // Center lock plate
        fillrect(cx+11, cy+8,   6,  6, RGBA32(215, 175,  0, 255));
        // Keyhole
        fillrect(cx+13, cy+9,   2,  2, RGBA32(30,  20,   5, 255));
        fillrect(cx+13, cy+11,  2,  3, RGBA32(30,  20,   5, 255));
        // Shimmer glow on lid — pulses every ~30 frames
        int glow = ((f / 30) & 1) ? 60 : 20;
        fillrect(cx+3,  cy+3,   8,  2, RGBA32(255, 230, 100, glow));
        fillrect(cx+14, cy+3,   8,  2, RGBA32(255, 230, 100, glow));
    }

    // ── Attack slash arc + impact sparks (driven by G.swing_timer) ─────────
    if (G.swing_timer > 0) {
        int prog = SWING_FRAMES - G.swing_timer;  /* 0..19 */
        /* Anchor at sword tip, project forward in facing direction */
        int tip_x = sx + sw_off, tip_y = sy - 14;
        int fwd_x = sx, fwd_y = sy;
        switch (G.sophia_face) {
            case FACE_UP:    fwd_x = sx;      fwd_y = sy - 28; break;
            case FACE_DOWN:  fwd_x = sx;      fwd_y = sy + 28; break;
            case FACE_LEFT:  fwd_x = sx - 28; fwd_y = sy;      break;
            case FACE_RIGHT: fwd_x = sx + 28; fwd_y = sy;      break;
        }
        int denom = (SWING_FRAMES - 1);
        int lx = tip_x + ((fwd_x - tip_x) * prog) / denom;
        int ly = tip_y + ((fwd_y - tip_y) * prog) / denom;
        /* Slash phase: bright arc */
        if (prog <= 14) {
            fillrect(lx-2, ly-2, 6, 6, RGBA32(255, 255, 120, 255));
            fillrect(lx-1, ly-1, 4, 4, RGBA32(255, 240, 60,  255));
            if (prog > 2) {
                int lx2 = tip_x + ((fwd_x - tip_x) * (prog - 3)) / denom;
                int ly2 = tip_y + ((fwd_y - tip_y) * (prog - 3)) / denom;
                fillrect(lx2-1, ly2-1, 4, 4, RGBA32(220, 200, 40, 180));
            }
        } else {
            /* Follow-through fade */
            fillrect(lx-1, ly-1, 3, 3, RGBA32(180, 140, 20, 120));
        }
    }
    /* Hit-flash sparks at wounded enemies */
    for (int i = 0; i < MAX_ENEMIES; i++) {
        const Enemy *e = &G.enemies[i];
        if (e->kind == ENEMY_NONE || e->hurt_timer <= 0) continue;
        int sp = 12 - e->hurt_timer;  /* grows 0..11 during flash */
        if (sp < 0) sp = 0;
        int hx2 = e->x, hy2 = e->y - (e->kind == ENEMY_STALFOS ? 10 : 0);
        fillrect(hx2 + sp,   hy2 - sp/2, 3, 3, RGBA32(255, 230,   0, 255));
        fillrect(hx2 - sp,   hy2 + sp/2, 3, 3, RGBA32(255, 200,  50, 255));
        if (sp < 5) fillrect(hx2 - 3, hy2 - 3, 7, 7, RGBA32(255, 255, 200, 255));
    }
    /* Unused-var suppression for legacy atk (still decremented above) */
    (void)atk;

    // ── HUD: 4 Hearts + Magic Bar (drawn last = on top) ─────────────────────
    fillrect(0, 0, 320, 14, RGBA32(0, 0, 0, 255));   // dark HUD band

    // 4 heart containers — each 8×6 px, gap of 2px → 10px per heart
    for (int h = 0; h < 4; h++) {
        int hx = 4 + h * 12;
        int hy = 3;
        // Full heart = 2 half-hearts. G.hearts tracks half-hearts.
        color_t hcol = (G.hearts >= (h * 2 + 2)) ? RGBA32(220,  30,  30, 255) :
                       (G.hearts == (h * 2 + 1)) ? RGBA32(220,  30,  30, 255) :
                                                    RGBA32( 60,  12,  12, 255);
        // Heart shape: two bumps + wide body + narrowing point
        fillrect(hx+1, hy,   2, 2, hcol);   // left bump
        fillrect(hx+5, hy,   2, 2, hcol);   // right bump
        fillrect(hx,   hy+1, 8, 3, hcol);   // middle body
        fillrect(hx+1, hy+4, 6, 1, hcol);   // taper 1
        fillrect(hx+2, hy+5, 4, 1, hcol);   // taper 2
        fillrect(hx+3, hy+6, 2, 1, hcol);   // tip
        // Half-heart: overlay right half dark when half-full
        if (G.hearts == (h * 2 + 1)) {
            color_t hdim = RGBA32(60, 12, 12, 255);
            fillrect(hx+4, hy,   4, 2, hdim);
            fillrect(hx+4, hy+1, 4, 3, hdim);
            fillrect(hx+4, hy+4, 3, 1, hdim);
            fillrect(hx+5, hy+5, 2, 1, hdim);
        }
    }

    // Rupee icon + counter (center HUD)
    {
        int rx = 116, ry = 3;
        /* Green rupee pictogram */
        color_t rbase = RGBA32( 60, 200, 110, 255);
        color_t rhi   = RGBA32(180, 255, 200, 255);
        fillrect(rx+2, ry,   2, 1, rbase);
        fillrect(rx+1, ry+1, 4, 1, rbase);
        fillrect(rx,   ry+2, 6, 3, rbase);
        fillrect(rx+1, ry+5, 4, 1, rbase);
        fillrect(rx+2, ry+6, 2, 1, rbase);
        fillrect(rx+2, ry+2, 1, 2, rhi);
        /* x-000 digit readout drawn via graphics_draw_text in CPU pass
         * (handled in draw_text STATE_DUNGEON) */
    }

    // Kills counter icon (tiny skull, right of rupees)
    {
        int kx_hud = 156, ky_hud = 3;
        color_t skull = RGBA32(220,220,200,255);
        color_t eye   = RGBA32(20,10,20,255);
        fillrect(kx_hud,   ky_hud,   6, 5, skull);
        fillrect(kx_hud+1, ky_hud+1, 1, 1, eye);
        fillrect(kx_hud+4, ky_hud+1, 1, 1, eye);
        fillrect(kx_hud+1, ky_hud+5, 4, 1, skull);
        fillrect(kx_hud+1, ky_hud+6, 1, 1, skull);
        fillrect(kx_hud+4, ky_hud+6, 1, 1, skull);
    }

    // Magic bar — green gradient, right side of HUD
    {
        int bx = 200, by = 4;
        fillrect(bx,    by,    68,  6, RGBA32( 10,  10,  30, 255)); // background
        int fill = (G.magic * 64) / 128;                             // 0-64 px
        if (fill > 0) {
            fillrect(bx+2, by+1, fill, 2, RGBA32( 80, 255, 130, 255)); // top highlight
            fillrect(bx+2, by+3, fill, 2, RGBA32( 40, 200,  90, 255)); // lower fill
        }
        fillrect(bx,    by,    68,  1, RGBA32( 80, 180,  80, 255)); // top border
        fillrect(bx,    by+5,  68,  1, RGBA32( 80, 180,  80, 255)); // bottom border
        fillrect(bx,    by,     1,  6, RGBA32( 80, 180,  80, 255)); // left border
        fillrect(bx+67, by,     1,  6, RGBA32( 80, 180,  80, 255)); // right border
    }

    // Performance bars (RDP-drawn during generation)
    if (G.perf_show && (G.state == STATE_GENERATING || G.state == STATE_DIALOG)) {
        int bar_y = 140;

        // CPU bar background
        fillrect(28, bar_y, 84, 6, RGBA32(20, 20, 20, 255));
        // CPU bar fill (red→orange gradient based on load)
        {
            int cpu_fill = (int)(G.perf_cpu_pct * 0.80f);
            if (cpu_fill > 80) cpu_fill = 80;
            if (cpu_fill < 1) cpu_fill = 1;
            if (G.perf_cpu_pct > 80.0f) {
                fillrect(30, bar_y+1, cpu_fill, 4, RGBA32(255, 60, 30, 255));  // red = heavy
            } else if (G.perf_cpu_pct > 40.0f) {
                fillrect(30, bar_y+1, cpu_fill, 4, RGBA32(255, 160, 30, 255)); // orange
            } else {
                fillrect(30, bar_y+1, cpu_fill, 4, RGBA32(80, 220, 80, 255));  // green = light
            }
        }
        // CPU bar border
        fillrect(28, bar_y,   84, 1, RGBA32(100, 100, 100, 255));
        fillrect(28, bar_y+5, 84, 1, RGBA32(100, 100, 100, 255));
        fillrect(28, bar_y,    1, 6, RGBA32(100, 100, 100, 255));
        fillrect(111, bar_y,   1, 6, RGBA32(100, 100, 100, 255));

#ifdef USE_RSP_MATMUL
        // RSP bar (always shows "active" during generation since RSP does matmul)
        fillrect(130, bar_y, 84, 6, RGBA32(20, 20, 20, 255));
        if (G.state == STATE_GENERATING) {
            fillrect(132, bar_y+1, 60, 4, RGBA32(30, 200, 255, 255));  // cyan = RSP active
        }
        fillrect(130, bar_y,   84, 1, RGBA32(100, 100, 100, 255));
        fillrect(130, bar_y+5, 84, 1, RGBA32(100, 100, 100, 255));
        fillrect(130, bar_y,    1, 6, RGBA32(100, 100, 100, 255));
        fillrect(213, bar_y,   1, 6, RGBA32(100, 100, 100, 255));
#endif
#ifdef USE_RPC_LLM
        if (G.rpc_active) {
            // RPC bar (purple = remote inference)
            fillrect(130, bar_y, 84, 6, RGBA32(20, 20, 20, 255));
            if (G.state == STATE_GENERATING && G.rpc_pending) {
                // Pulsing bar while waiting for RPC response
                int pulse = (G.frame / 4) % 60;
                fillrect(132, bar_y+1, pulse + 20, 4, RGBA32(180, 80, 255, 255));
            } else if (G.perf_gen_total_us > 0) {
                // Solid bar showing latency quality after response
                // 200ms = full (80px), 2000ms = minimum (8px)
                int latency_ms = (int)(G.perf_gen_total_us / 1000);
                int rpc_fill = 80 - (latency_ms - 200) / 25;
                if (rpc_fill > 80) rpc_fill = 80;
                if (rpc_fill < 8)  rpc_fill = 8;
                fillrect(132, bar_y+1, rpc_fill, 4, RGBA32(140, 60, 220, 255));
            }
            // Purple border
            fillrect(130, bar_y,   84, 1, RGBA32(140, 80, 180, 255));
            fillrect(130, bar_y+5, 84, 1, RGBA32(140, 80, 180, 255));
            fillrect(130, bar_y,    1, 6, RGBA32(140, 80, 180, 255));
            fillrect(213, bar_y,   1, 6, RGBA32(140, 80, 180, 255));
        }
#endif

        // Tok/s numeric right-aligned
        {
            int whole = (int)G.perf_toks_precise;
            int frac  = (int)((G.perf_toks_precise - (float)whole) * 10.0f);
            if (whole > 99) { whole = 99; frac = 9; }
            char tsbuf[12];
            int i = 0;
            if (whole >= 10) tsbuf[i++] = '0' + whole / 10;
            tsbuf[i++] = '0' + whole % 10;
            tsbuf[i++] = '.';
            tsbuf[i++] = '0' + frac;
            tsbuf[i++] = ' ';
            tsbuf[i++] = 't';
            tsbuf[i++] = 'o';
            tsbuf[i++] = 'k';
            tsbuf[i++] = '/';
            tsbuf[i++] = 's';
            tsbuf[i] = '\0';
            // Will be drawn with graphics_draw_text in SW pass
        }
    }

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

// ─── Virtual Keyboard ────────────────────────────────────────────────────────
// 4 rows x 10 cols D-pad character picker for player text input to Sophia

static const char KB_GRID[4][11] = {
    "ABCDEFGHIJ",
    "KLMNOPQRST",
    "UVWXYZ .,?",
    "0123456789",
};

static void scene_keyboard(void) {
    /* Dark background */
    fillrect(0, 0, 320, 240, RGBA32(10, 5, 20, 255));

    /* Input display area (top) */
    fillrect(8, 8, 304, 24, RGBA32(0, 0, 40, 255));
    fillrect(8, 8, 304, 1, RGBA32(215,175,0,255));
    fillrect(8, 31, 304, 1, RGBA32(215,175,0,255));
    fillrect(8, 8, 1, 24, RGBA32(215,175,0,255));
    fillrect(311, 8, 1, 24, RGBA32(215,175,0,255));

    /* Keyboard grid background */
    fillrect(20, 80, 280, 120, RGBA32(0, 0, 50, 255));

    /* Highlight selected cell */
    int cx = 28 + G.kb_col * 28;
    int cy = 84 + G.kb_row * 28;
    fillrect(cx, cy, 24, 24, RGBA32(180, 140, 0, 255));

    /* Grid cell borders */
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 10; c++) {
            int x = 28 + c * 28;
            int y = 84 + r * 28;
            fillrect(x, y, 24, 1, RGBA32(80, 60, 120, 255));
            fillrect(x, y+23, 24, 1, RGBA32(80, 60, 120, 255));
            fillrect(x, y, 1, 24, RGBA32(80, 60, 120, 255));
            fillrect(x+23, y, 1, 24, RGBA32(80, 60, 120, 255));
        }
    }

    /* Bottom bar */
    fillrect(20, 210, 280, 1, RGBA32(100, 80, 20, 255));
}

static void draw_keyboard_text(surface_t *disp) {
    graphics_draw_text(disp, 16, 2, "Ask Sophia:");

    /* Show current input with cursor */
    char ibuf[66];
    int len = G.kb_len;
    if (len > 60) len = 60;
    memcpy(ibuf, G.kb_input, len);
    ibuf[len] = '_';
    ibuf[len + 1] = '\0';
    graphics_draw_text(disp, 14, 14, ibuf);

    /* Draw keyboard grid characters */
    char cbuf[2] = {0, 0};
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 10; c++) {
            cbuf[0] = KB_GRID[r][c];
            int x = 36 + c * 28;
            int y = 90 + r * 28;
            graphics_draw_text(disp, x, y, cbuf);
        }
    }

    /* Instructions */
    graphics_draw_text(disp, 20, 216, "[A]Select [B]Delete [START]Send");
    /* Show char count */
    char cntbuf[8];
    cntbuf[0] = '0' + (G.kb_len / 10) % 10;
    cntbuf[1] = '0' + G.kb_len % 10;
    cntbuf[2] = '/';
    cntbuf[3] = '6';
    cntbuf[4] = '0';
    cntbuf[5] = '\0';
    graphics_draw_text(disp, 270, 2, cntbuf);
}

static void handle_keyboard_input(struct controller_data *k) {
    if (G.kb_debounce > 0) { G.kb_debounce--; return; }

    /* D-pad navigation */
    if (k->c[0].up)    { G.kb_row = (G.kb_row + 3) % 4; G.kb_debounce = 5; }
    if (k->c[0].down)  { G.kb_row = (G.kb_row + 1) % 4; G.kb_debounce = 5; }
    if (k->c[0].left)  { G.kb_col = (G.kb_col + 9) % 10; G.kb_debounce = 5; }
    if (k->c[0].right) { G.kb_col = (G.kb_col + 1) % 10; G.kb_debounce = 5; }

    /* A = select character */
    if (k->c[0].A && G.kb_len < 60) {
        char ch = KB_GRID[G.kb_row][G.kb_col];
        /* Convert to lowercase for model input */
        if (ch >= 'A' && ch <= 'Z') ch = ch + 32;
        G.kb_input[G.kb_len++] = ch;
        G.kb_debounce = 8;
    }

    /* B = backspace */
    if (k->c[0].B && G.kb_len > 0) {
        G.kb_len--;
        G.kb_input[G.kb_len] = '\0';
        G.kb_debounce = 8;
    }

    /* START = send to Sophia */
    if (k->c[0].start && G.kb_len > 0) {
        /* Transition to generating with player's custom prompt */
        G.state = STATE_GENERATING;
        G.dialog_char = 0;
        G.dialog_done = 0;
        G.dialog_len  = 0;
        G.gen_out_count   = 0;
        G.gen_start_frame = G.frame;
        G.gen_toks_sec    = 0.0f;
        G.perf_gen_cycles   = 0;
        G.perf_gen_total_us = 0;
        G.perf_gen_start_us = CYCLES_TO_US(TICKS_READ());
        G.perf_cpu_pct      = 0.0f;
        G.perf_toks_precise = 0.0f;
        G.perf_show         = 1;
#ifdef USE_RPC_LLM
        G.rpc_pending       = 0;
#endif
        memset(G.dialog_buf, 0, sizeof(G.dialog_buf));

        if (G.ai_ready) {
            sgai_reset(&G.ai);
            int plen = G.kb_len;
            if (plen > (int)sizeof(G.gen_pbuf) - 1)
                plen = (int)sizeof(G.gen_pbuf) - 1;
            memcpy(G.gen_pbuf, G.kb_input, plen);
            G.gen_plen     = plen;
            G.gen_ppos     = 0;
            G.gen_last_tok = G.gen_pbuf[0];
        } else {
            strncpy((char *)G.dialog_buf, "I hear you, friend!", sizeof(G.dialog_buf) - 1);
            G.dialog_len = 19;
            G.gen_plen   = 0;
            G.gen_ppos   = 0;
        }

        /* Clear keyboard buffer for next time */
        G.kb_len = 0;
        memset(G.kb_input, 0, sizeof(G.kb_input));
    }

    /* Z = cancel, back to dungeon */
    if (k->c[0].Z) {
        G.state = STATE_DUNGEON;
        G.kb_debounce = 10;
    }
}

// ─── CPU text overlay ────────────────────────────────────────────────────────

static void draw_text(surface_t *disp) {
    switch (G.state) {

    case STATE_ANNIVERSARY:
        graphics_draw_text(disp,  64, 72,  "World's First N64 LLM");
        graphics_draw_text(disp,  80, 86,  "Legend of Elya");
        graphics_draw_text(disp,  80, 100, "Elyan Labs  2026");
        graphics_draw_text(disp,  96, 168, "from Elyan Labs");
        graphics_draw_text(disp,  68, 182, "World's First N64 LLM");
        if ((G.frame / 20) & 1)
            graphics_draw_text(disp, 80, 220, "Press START to continue");
        break;

    case STATE_TITLE:
        graphics_draw_text(disp, 104, 50, "LEGEND OF ELYA");
        graphics_draw_text(disp,  80, 68, "Nintendo 64 Homebrew");
        graphics_draw_text(disp, 120, 84, "Elyan Labs");
        graphics_draw_text(disp,  76,103, "World's First N64 LLM");
        if (G.ai_ready && G.ai.is_loaded)
            graphics_draw_text(disp,  84, 118, "[Sophia AI: LOADED]");
        else if (G.ai_ready)
            graphics_draw_text(disp,  64, 118, "[AI: file ok, magic?]");
        else
            graphics_draw_text(disp,  72, 118, "[Sophia AI: Demo Mode]");
        graphics_draw_text(disp,  80, 155, "Press START to enter");
        graphics_draw_text(disp, 104, 170, "the dungeon...");
        break;

    case STATE_DUNGEON: {
        graphics_draw_text(disp, 186, 3,  "MP");   // magic bar label
        /* Rupee count: right of rupee icon, 3-digit pad */
        {
            char rbuf[8];
            int r = G.rupees; if (r > 999) r = 999;
            rbuf[0] = 'x';
            rbuf[1] = '0' + (r / 100) % 10;
            rbuf[2] = '0' + (r /  10) % 10;
            rbuf[3] = '0' +  r        % 10;
            rbuf[4] = '\0';
            graphics_draw_text(disp, 126, 3, rbuf);
        }
        /* Kills count: right of skull icon, 2-digit */
        {
            char kbuf[6];
            int k2 = G.kills; if (k2 > 99) k2 = 99;
            kbuf[0] = 'x';
            kbuf[1] = '0' + (k2 / 10) % 10;
            kbuf[2] = '0' +  k2       % 10;
            kbuf[3] = '\0';
            graphics_draw_text(disp, 166, 3, kbuf);
        }
        /* Room indicator (right edge of HUD) */
        {
            char rmbuf[6];
            rmbuf[0] = 'R';
            rmbuf[1] = '0' + (G.room + 1);
            rmbuf[2] = '/';
            rmbuf[3] = '0' + N_ROOMS;
            rmbuf[4] = '\0';
            graphics_draw_text(disp, 288, 3, rmbuf);
        }
        /* Door hint when room is cleared and there's somewhere to go east */
        if (G.room_cleared[G.room] && G.room < N_ROOMS - 1
            && G.sophia_x > 240 && ((G.frame / 20) & 1))
            graphics_draw_text(disp, 260, 132, ">>");
        if (G.room > 0 && G.sophia_x < 80 && ((G.frame / 20) & 1))
            graphics_draw_text(disp,  40, 132, "<<");

        /* Control hint — flips between two messages so all 4 bindings land */
        if ((G.frame / 180) & 1)
            graphics_draw_text(disp, 4, 220, "[stick]Move [A]Attack/Talk [B]Type [Z]Perf");
        else if (!G.room_cleared[G.room])
            graphics_draw_text(disp, 4, 220, "Clear the room to open the east door.");
        else
            graphics_draw_text(disp, 4, 220, "The way east is open.  Press A near Sophia to talk.");
        break;
    }

    case STATE_TRANSITION:
        /* Silent — backdrop handles visuals */
        break;

    case STATE_VICTORY: {
        int since = G.frame - G.victory_frame;
        graphics_draw_text(disp, 100, 60, "* VICTORY *");
        graphics_draw_text(disp,  60, 76, "The dungeon's curse is lifted.");
        graphics_draw_text(disp,  40, 110, "Sophia Elya stands triumphant.");
        char srupees[24];
        int r = G.rupees; if (r > 999) r = 999;
        int k2 = G.kills; if (k2 > 99) k2 = 99;
        srupees[0]='R'; srupees[1]='u'; srupees[2]='p'; srupees[3]='e';
        srupees[4]='e'; srupees[5]='s'; srupees[6]=':'; srupees[7]=' ';
        srupees[8]  = '0' + (r / 100) % 10;
        srupees[9]  = '0' + (r /  10) % 10;
        srupees[10] = '0' +  r        % 10;
        srupees[11] = '\0';
        graphics_draw_text(disp, 112, 138, srupees);
        char skills[16];
        skills[0]='K'; skills[1]='i'; skills[2]='l'; skills[3]='l';
        skills[4]='s'; skills[5]=':'; skills[6]=' ';
        skills[7] = '0' + (k2 / 10) % 10;
        skills[8] = '0' +  k2       % 10;
        skills[9] = '\0';
        graphics_draw_text(disp, 112, 152, skills);
        if (since > 120 && ((G.frame / 30) & 1))
            graphics_draw_text(disp,  84, 200, "Press START to replay");
        break;
    }

    case STATE_GAMEOVER: {
        /* Full-screen dark overlay handled in RDP pass below; text here */
        int since = G.frame - G.game_over_frame;
        graphics_draw_text(disp, 112, 60, "* GAME OVER *");
        graphics_draw_text(disp,  92, 82, "The dungeon claims you...");
        char scorebuf[32];
        int r = G.rupees; if (r > 999) r = 999;
        int k2 = G.kills; if (k2 > 99) k2 = 99;
        scorebuf[0] = 'R'; scorebuf[1] = 'u'; scorebuf[2] = 'p'; scorebuf[3] = 'e';
        scorebuf[4] = 'e'; scorebuf[5] = 's'; scorebuf[6] = ':'; scorebuf[7] = ' ';
        scorebuf[8]  = '0' + (r / 100) % 10;
        scorebuf[9]  = '0' + (r /  10) % 10;
        scorebuf[10] = '0' +  r        % 10;
        scorebuf[11] = '\0';
        graphics_draw_text(disp, 112, 108, scorebuf);
        char killbuf[24];
        killbuf[0] = 'K'; killbuf[1] = 'i'; killbuf[2] = 'l'; killbuf[3] = 'l';
        killbuf[4] = 's'; killbuf[5] = ':'; killbuf[6] = ' ';
        killbuf[7] = '0' + (k2 / 10) % 10;
        killbuf[8] = '0' +  k2       % 10;
        killbuf[9] = '\0';
        graphics_draw_text(disp, 112, 122, killbuf);
        if (since > 60 && ((G.frame / 30) & 1))
            graphics_draw_text(disp, 80, 170, "Press START to try again");
        break;
    }

    case STATE_KEYBOARD:
        draw_keyboard_text(disp);
        break;

    case STATE_DIALOG:
    case STATE_GENERATING: {
        graphics_draw_text(disp, 16, 158, "Sophia Elya:");

        // Performance overlay during generation
        if (G.state == STATE_GENERATING && G.gen_out_count > 0) {
            // Tok/s (precise)
            char spdbuf[16];
            int whole = (int)G.perf_toks_precise;
            int frac  = (int)((G.perf_toks_precise - (float)whole) * 10.0f);
            if (whole > 99) whole = 99;
            spdbuf[0] = (whole >= 10) ? ('0' + whole / 10) : ' ';
            spdbuf[1] = '0' + (whole % 10);
            spdbuf[2] = '.';
            spdbuf[3] = '0' + frac;
            spdbuf[4] = ' ';
            spdbuf[5] = 't';
            spdbuf[6] = '/';
            spdbuf[7] = 's';
            spdbuf[8] = '\0';
            graphics_draw_text(disp, 200, 158, spdbuf);

            // Total gen time in ms
            {
                int ms = (int)(G.perf_gen_total_us / 1000);
                char tbuf[12];
                int ti = 0;
                if (ms >= 1000) { tbuf[ti++] = '0' + (ms / 1000) % 10; ms %= 1000; }
                if (ms >= 100 || ti > 0) tbuf[ti++] = '0' + (ms / 100) % 10;
                tbuf[ti++] = '0' + (ms / 10) % 10;
                tbuf[ti++] = '0' + ms % 10;
                tbuf[ti++] = 'm'; tbuf[ti++] = 's'; tbuf[ti] = '\0';
                graphics_draw_text(disp, 260, 158, tbuf);
            }
        }

        // Performance bars (show during and briefly after generation)
        if (G.perf_show && (G.state == STATE_GENERATING || G.perf_cpu_pct > 0.0f)) {
            int bar_y = 145;  // just above dialog box

            // CPU inference bar (red/orange)
            {
                int cpu_fill = (int)(G.perf_cpu_pct * 0.80f);  // 80px max
                if (cpu_fill > 80) cpu_fill = 80;
                if (cpu_fill < 0) cpu_fill = 0;
                graphics_draw_text(disp, 4, bar_y - 1, "CPU");
            }
#ifdef USE_RSP_MATMUL
            // RSP indicator (cyan)
            graphics_draw_text(disp, 44, bar_y - 1, "RSP");
#endif
#ifdef USE_RPC_LLM
            if (G.rpc_active) {
                // RPC label and latency display
                graphics_draw_text(disp, 44, bar_y - 1, "RPC");
                // Show RPC latency in ms when available
                if (G.perf_gen_total_us > 0 && G.state == STATE_DIALOG) {
                    char rpcbuf[12];
                    int ms = (int)(G.perf_gen_total_us / 1000);
                    int ri = 0;
                    if (ms >= 1000) { rpcbuf[ri++] = '0' + (ms/1000)%10; }
                    if (ms >= 100 || ri > 0) rpcbuf[ri++] = '0' + (ms/100)%10;
                    rpcbuf[ri++] = '0' + (ms/10)%10;
                    rpcbuf[ri++] = '0' + ms%10;
                    rpcbuf[ri++] = 'm'; rpcbuf[ri++] = 's'; rpcbuf[ri] = '\0';
                    graphics_draw_text(disp, 82, bar_y - 1, rpcbuf);
                } else if (G.rpc_pending) {
                    // Pulsing dots while waiting
                    int dots = (G.frame / 10) % 4;
                    char dotbuf[5] = "    ";
                    for (int d = 0; d < dots; d++) dotbuf[d] = '.';
                    dotbuf[dots] = '\0';
                    graphics_draw_text(disp, 82, bar_y - 1, dotbuf);
                }
            }
#endif
        }

        // Character reveal with word-wrap
        int show = (G.dialog_char < 90) ? G.dialog_char : 90;
        char linebuf[37];
        int lb = 0, col = 0, line_y = 174;

        for (int i = 0; i < show; i++) {
            unsigned char c = G.dialog_buf[i];
            if (c < 32 || c > 126) continue;  /* skip any residual non-printable */
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

// ─── Per-frame generation (one token per frame) ───────────────────────────────

static void update_generating_step(void) {
#ifdef USE_RPC_LLM
    /* RPC path: send prompt to bridge, poll for response */
    if (G.rpc_active) {
        if (!G.rpc_pending && G.gen_plen > 0) {
            /* Build prompt string from gen_pbuf */
            char prompt_str[65];
            int plen = G.gen_plen;
            if (plen > 64) plen = 64;
            for (int i = 0; i < plen; i++)
                prompt_str[i] = (char)G.gen_pbuf[i];
            prompt_str[plen] = '\0';

            /* Send RPC request: max 80 tokens, temp=0.25 (Q8=64) */
            if (llm_rpc_request(prompt_str, 80, 64)) {
                G.rpc_pending = 1;
                G.rpc_send_us = CYCLES_TO_US(TICKS_READ());
                debugf("RPC sent: '%s'\n", prompt_str);
            }
        }

        if (G.rpc_pending) {
            int status = llm_rpc_poll();
            if (status == LLM_STATUS_READY) {
                /* Got response! Copy to dialog buffer */
                int rlen = g_llm_rpc.response_len;
                if (rlen > 80) rlen = 80;
                for (int i = 0; i < rlen; i++) {
                    uint8_t c = g_llm_rpc.response_buf[i];
                    if (c >= 32 && c <= 126)
                        G.dialog_buf[G.dialog_len++] = c;
                    if (c == '\n' || c == '\0') break;
                }
                G.dialog_char = G.dialog_len;

                /* Compute RPC latency */
                uint32_t now_us = CYCLES_TO_US(TICKS_READ());
                G.perf_gen_total_us = now_us - G.rpc_send_us;
                if (G.perf_gen_total_us > 1000) {
                    G.perf_toks_precise = (float)G.dialog_len * 1000000.0f / (float)G.perf_gen_total_us;
                    G.gen_toks_sec = G.perf_toks_precise;
                }
                G.perf_cpu_pct = 5.0f;  /* RPC = minimal CPU usage */

                filter_dialog_buf();
                G.dialog_done = 1;
                G.state = STATE_DIALOG;
                G.rpc_pending = 0;
                debugf("RPC response (%d us): '%s'\n",
                       (int)G.perf_gen_total_us, G.dialog_buf);
                return;
            } else if (status == LLM_STATUS_ERROR) {
                /* RPC failed, fall through to local inference */
                G.rpc_pending = 0;
                G.rpc_active = 0;  /* Disable RPC, use local from now on */
                debugf("RPC error — falling back to local LLM\n");
            }
            /* Still pending — show waiting animation */
            if (G.rpc_pending) {
                G.gen_out_count++;
                G.perf_cpu_pct = 2.0f;
                return;
            }
        }
    }
#endif

    if (!G.ai_ready) {
        // Canned mode: reveal one character every other frame
        if ((G.frame & 1) == 0 && G.dialog_char < G.dialog_len)
            G.dialog_char++;
        if (G.dialog_char >= G.dialog_len) {
            G.dialog_done = 1;
            G.state = STATE_DIALOG;
        }
        return;
    }

    if (G.gen_ppos < G.gen_plen) {
        // Phase 0: feed prompt tokens (discard output; temperature=0)
        {
            uint32_t _t0 = TICKS_READ();
            G.gen_last_tok = sgai_next_token(&G.ai,
                                              G.gen_pbuf[G.gen_ppos], 0);
            G.perf_gen_cycles += TICKS_READ() - _t0;
        }
        G.gen_ppos++;
        if (G.gen_ppos >= G.gen_plen) {
            // Prompt fully fed — gen_last_tok now holds the model's prediction
            // from the last prompt token (greedy argmax, printable ASCII).
            // DO NOT overwrite it — that prediction seeds the first output token.
            G.gen_start_frame = G.frame;
            G.gen_out_count   = 0;
        }
    } else {
        // Phase 1: generate one output token
        // temp_q8=64 → T=0.25 (mild randomness — varied but coherent outputs for demo)
        uint32_t _t0 = TICKS_READ();
        uint8_t tok = sgai_next_token(&G.ai, G.gen_last_tok, 64);
        G.perf_gen_cycles += TICKS_READ() - _t0;
        G.gen_last_tok = tok;
        G.gen_out_count++;

        // Newline = end of Q&A response (training separator); treat like EOS
        if (tok == '\n') tok = 0;
        // Period = end of first sentence — stop here for a clean response.
        // Every training answer ends with "." so the model reliably emits one.
        // Require 8+ output chars first to skip any period inside abbreviations.
        if (tok == '.' && G.gen_out_count >= 8) tok = 0;

        // Append token — sample_logits already restricts to printable ASCII 32-126,
        // but double-check here as defensive measure (unsigned char cast matters)
        if (tok != 0 && (unsigned char)tok >= 32 && (unsigned char)tok <= 126
            && G.dialog_len < (int)sizeof(G.dialog_buf) - 1) {
            G.dialog_buf[G.dialog_len++] = tok;
            G.dialog_char = G.dialog_len;   // show immediately
        }

        // Update tok/s — precise using CP0 cycle counter
        {
            uint32_t now_us = CYCLES_TO_US(TICKS_READ());
            uint32_t elapsed_us = now_us - G.perf_gen_start_us;
            if (elapsed_us > 1000) {  // at least 1ms elapsed
                G.perf_toks_precise = (float)G.gen_out_count * 1000000.0f / (float)elapsed_us;
                G.gen_toks_sec = G.perf_toks_precise;
                G.perf_gen_total_us = elapsed_us;
            }
            // CPU% = inference cycles / frame cycles
            // Average over frames since gen started
            int frames_elapsed = G.frame - G.gen_start_frame;
            if (frames_elapsed > 0) {
                uint32_t total_frame_cycles = (uint32_t)frames_elapsed * FRAME_CYCLES;
                // perf_gen_cycles accumulates across all frames
                G.perf_cpu_pct = (float)G.perf_gen_cycles * 100.0f / (float)total_frame_cycles;
                if (G.perf_cpu_pct > 100.0f) G.perf_cpu_pct = 100.0f;
            }
        }

        // Stop when null/newline token, max output, or buffer full
        if (tok == 0 || G.dialog_len >= 80) {
            filter_dialog_buf();   /* strip training artifacts (helpmeet etc.) */
            G.dialog_done = 1;
            G.state = STATE_DIALOG;
        }
    }
}

// ─── Dialog logic ─────────────────────────────────────────────────────────────

static void start_dialog(void) {
    G.state       = STATE_GENERATING;
    G.dialog_char = 0;
    G.dialog_done = 0;
    G.dialog_len  = 0;
    G.gen_out_count   = 0;
    G.gen_start_frame = G.frame;
    G.gen_toks_sec    = 0.0f;
    G.perf_gen_cycles   = 0;
    G.perf_gen_total_us = 0;
    G.perf_gen_start_us = CYCLES_TO_US(TICKS_READ());
    G.perf_cpu_pct      = 0.0f;
    G.perf_toks_precise = 0.0f;
    G.perf_show         = 1;
#ifdef USE_RPC_LLM
    G.rpc_pending       = 0;
#endif
    memset(G.dialog_buf, 0, sizeof(G.dialog_buf));

    /* Hardware entropy seed selection — RIP-PoA oscillator trick:
     * XOR CPU cycle counter low bits with frame + last token hash.
     * Low bits of TICKS_READ() vary ~12 bits per A-press due to
     * player reaction time jitter and music sample phase offset.
     * This guarantees a different prompt is chosen nearly every time. */
    uint32_t entropy = N64_ENTROPY();
    int idx = (int)(entropy % N_PROMPTS);
    G.prompt_idx++;   /* also increment for sequential tracking */

    if (G.ai_ready) {
        sgai_reset(&G.ai);
        const char *p = PROMPTS[idx];
        int plen = (int)strlen(p);
        /* Feed bare prompt — no seed prefix.
         * Context=32 tokens; seeds were burning 13-14 tokens leaving <5 for
         * response. Bare prompt (13-20 chars) leaves 12-19 response tokens. */
        if (plen > (int)sizeof(G.gen_pbuf) - 1)
            plen = (int)sizeof(G.gen_pbuf) - 1;
        memcpy(G.gen_pbuf, p, plen);
        G.gen_plen     = plen;
        G.gen_ppos     = 0;
        G.gen_last_tok = G.gen_pbuf[0];
    } else {
        // Canned fallback: entropy-selected from N_CANNED pool
        const char *resp = CANNED[idx % N_CANNED];
        strncpy((char *)G.dialog_buf, resp, sizeof(G.dialog_buf) - 1);
        G.dialog_len = (int)strlen(resp);
        G.gen_plen   = 0;   // signals canned path in update_generating_step
        G.gen_ppos   = 0;
    }
}

// ─── Input ────────────────────────────────────────────────────────────────────

static void handle_input(void) {
    controller_scan();
    struct controller_data k    = get_keys_down();
    struct controller_data held = get_keys_held();
    switch (G.state) {
    case STATE_ANNIVERSARY:
        if (k.c[0].start || k.c[0].A) G.state = STATE_TITLE;
        break;
    case STATE_TITLE:
        if (k.c[0].start || k.c[0].A) {
            reset_dungeon();
            G.state = STATE_DUNGEON;
        }
        break;
    case STATE_DUNGEON: {
        /* ── Movement: d-pad OR analog stick ──────────────────────────── */
        int mx = 0, my = 0;
        if (held.c[0].left)  mx -= SOPHIA_SPEED;
        if (held.c[0].right) mx += SOPHIA_SPEED;
        if (held.c[0].up)    my -= SOPHIA_SPEED;
        if (held.c[0].down)  my += SOPHIA_SPEED;
        /* Analog stick: dead zone 32, proportional up to speed 3 */
        int ax = held.c[0].x;
        int ay = held.c[0].y;
        if (ax >  32) mx =  SOPHIA_SPEED + (ax > 60 ? 1 : 0);
        if (ax < -32) mx = -SOPHIA_SPEED - (ax < -60 ? 1 : 0);
        if (ay >  32) my = -SOPHIA_SPEED - (ay > 60 ? 1 : 0);  /* stick up = y negative */
        if (ay < -32) my =  SOPHIA_SPEED + (ay < -60 ? 1 : 0);

        /* Freeze movement during swing to avoid slide-hit ambiguity */
        if (G.swing_timer > 0) { mx = 0; my = 0; }

        if (mx || my) {
            G.sophia_x += mx;
            G.sophia_y += my;
            if (G.sophia_x < SOPHIA_X_MIN) G.sophia_x = SOPHIA_X_MIN;
            if (G.sophia_x > SOPHIA_X_MAX) G.sophia_x = SOPHIA_X_MAX;
            if (G.sophia_y < SOPHIA_Y_MIN) G.sophia_y = SOPHIA_Y_MIN;
            if (G.sophia_y > SOPHIA_Y_MAX) G.sophia_y = SOPHIA_Y_MAX;
            /* Facing follows dominant axis, horizontal wins ties (feels right) */
            if (iabs(mx) >= iabs(my)) G.sophia_face = (mx < 0) ? FACE_LEFT : FACE_RIGHT;
            else                      G.sophia_face = (my < 0) ? FACE_UP   : FACE_DOWN;
        }

        /* ── A button: context-sensitive attack vs talk ───────────────── */
        if (k.c[0].A) {
            int tgt = find_enemy_in_range(G.sophia_x, G.sophia_y, ATTACK_RANGE);
            if (tgt >= 0 && G.swing_timer == 0) {
                /* Face the target before swinging */
                int dx = G.enemies[tgt].x - G.sophia_x;
                int dy = G.enemies[tgt].y - G.sophia_y;
                if (iabs(dx) >= iabs(dy)) G.sophia_face = (dx < 0) ? FACE_LEFT : FACE_RIGHT;
                else                      G.sophia_face = (dy < 0) ? FACE_UP   : FACE_DOWN;
                G.swing_timer    = SWING_FRAMES;
                G.swing_hit_done = 0;
                /* Legacy hit-flash system (scene_dungeon still uses these) */
                G.attack_timer   = 42;
                G.attack_target  = (G.enemies[tgt].kind == ENEMY_KEESE) ? 1 : 0;
            } else {
                /* No enemy in range → talk to Sophia (LLM) */
                start_dialog();
            }
        }
        /* ── B button: virtual keyboard for typed prompt ──────────────── */
        if (k.c[0].B) {
            G.state = STATE_KEYBOARD;
            G.kb_row = 0; G.kb_col = 0;
            G.kb_len = 0; G.kb_debounce = 10;
            memset(G.kb_input, 0, sizeof(G.kb_input));
        }
        /* ── Z: toggle performance overlay ────────────────────────────── */
        if (k.c[0].Z) G.perf_show = !G.perf_show;
        break;
    }
    case STATE_KEYBOARD:
        handle_keyboard_input(&k);
        break;
    case STATE_DIALOG:
        if (k.c[0].A) start_dialog();
        if (k.c[0].B) G.state = STATE_DUNGEON;
        break;
    case STATE_GENERATING:
        break;
    case STATE_GAMEOVER:
        /* Require ~1s fade-in before accepting retry input */
        if (G.frame - G.game_over_frame > 60 && (k.c[0].start || k.c[0].A)) {
            reset_dungeon();
            G.state = STATE_DUNGEON;
        }
        break;
    case STATE_VICTORY:
        if (G.frame - G.victory_frame > 120 && (k.c[0].start || k.c[0].A)) {
            reset_dungeon();
            G.state = STATE_DUNGEON;
        }
        break;
    case STATE_TRANSITION:
        /* Input locked during transition */
        break;
    }
}

// ─── Init ─────────────────────────────────────────────────────────────────────

static void game_init(void) {
    memset(&G, 0, sizeof(G));
    G.state = STATE_ANNIVERSARY;
    /* Record boot time via CP0 Count — increments at 46.875 MHz (half CPU clock).
     * Used for real-time splash duration, immune to emulator frame-rate variation. */
    uint32_t _cp0;
    asm volatile("mfc0 %0, $9" : "=r"(_cp0));
    G.anniversary_cp0 = _cp0;
    G.ai.kv  = &G.kv;
#ifdef USE_RPC_LLM
    /* Try to detect Pico bridge for RPC inference */
    G.rpc_active = llm_rpc_detect();
    if (G.rpc_active) {
        debugf("RPC bridge detected! Using remote inference.\n");
    } else {
        debugf("No RPC bridge — using on-cartridge LLM.\n");
    }
#endif

    reset_dungeon();  /* also sets hearts=8, magic=128, spawns initial enemies */

    int fd = dfs_open("/sophia_weights.bin");
    if (fd >= 0) {
        static uint8_t wbuf[1024 * 1024] __attribute__((aligned(8)));  /* 1MB for v5 Q8 4-layer */
        int sz = dfs_size(fd);
        if (sz > 0 && sz <= (int)sizeof(wbuf)) {
            dfs_read(wbuf, 1, sz, fd);
            dfs_close(fd);
            sgai_init(&G.ai, wbuf);
            G.ai.kv   = &G.kv;
            G.ai_ready = 1;
        } else {
            dfs_close(fd);
        }
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(void) {
    display_init(RESOLUTION_320x240, DEPTH_16_BPP, 2, GAMMA_NONE, ANTIALIAS_RESAMPLE);
    controller_init();
    timer_init();
    dfs_init(DFS_DEFAULT_LOCATION);
    rdpq_init();
    audio_init(MUSIC_FREQ, 4);   // 22kHz, 4 buffers for smooth square-wave

    game_init();

    while (1) {
        G.frame++;

        // Auto-advance anniversary screen after ~5 seconds using CP0 real-time clock.
        // 46,875,000 ticks = 1 second on N64. Immune to emulator frame-rate variation.
        if (G.state == STATE_ANNIVERSARY) {
            uint32_t _now;
            asm volatile("mfc0 %0, $9" : "=r"(_now));
            if ((_now - G.anniversary_cp0) >= 46875000u * 5u)
                G.state = STATE_TITLE;
        }

        // Per-frame AI generation step
        if (G.state == STATE_GENERATING)
            update_generating_step();

        // Per-frame gameplay (movement already applied in handle_input, this
        // advances enemy AI, collisions, pickups, swing hit-window, regen).
        update_world();

        // Get ONE surface for this frame
        surface_t *disp = display_get();

        // ── RDP graphics pass ──────────────────────────────────────────────
        G.perf_frame_start = TICKS_READ();
        rdpq_attach(disp, NULL);

        if (G.state == STATE_ANNIVERSARY) {
            scene_anniversary();
        } else if (G.state == STATE_TITLE) {
            fillrect(0, 0, 320, 240, RGBA32(0, 0, 20, 255));
            fillrect(30,  30, 260, 6, RGBA32(180, 140, 0, 255));
            fillrect(30, 130, 260, 6, RGBA32(180, 140, 0, 255));
        } else if (G.state == STATE_KEYBOARD) {
            scene_keyboard();
        } else if (G.state == STATE_GAMEOVER) {
            /* Fade-in backdrop: dark red vignette that deepens over ~1s */
            int since = G.frame - G.game_over_frame;
            int alpha = since * 4; if (alpha > 220) alpha = 220;
            fillrect(0, 0, 320, 240, RGBA32(0, 0, 0, 255));
            fillrect(0, 40, 320, 120, RGBA32(80, 0, 0, alpha));
        } else if (G.state == STATE_TRANSITION) {
            /* Fade out → swap → fade in. Render current room dim, with a
             * black curtain alpha scaled by distance-from-midpoint. */
            scene_dungeon();
            int half = TRANSITION_FRAMES / 2;
            int d = G.transition_frame <= half
                      ? G.transition_frame
                      : (TRANSITION_FRAMES - G.transition_frame);
            int alpha = (255 * (half - d)) / half;  /* 0 at edges, 255 at midpoint */
            if (alpha < 0) alpha = 0;
            if (alpha > 255) alpha = 255;
            fillrect(0, 0, 320, 240, RGBA32(0, 0, 0, alpha));
        } else if (G.state == STATE_VICTORY) {
            /* Gold-tinted congratulations screen */
            fillrect(0, 0, 320, 240, RGBA32(0, 0, 20, 255));
            int since = G.frame - G.victory_frame;
            int tw = (since / 2) % 12;
            /* Radiating gold bars */
            fillrect(0,  30 + tw, 320, 4, RGBA32(215, 175, 0, 255));
            fillrect(0, 200 - tw, 320, 4, RGBA32(215, 175, 0, 255));
            fillrect(30, 100, 260, 40, RGBA32(40, 30, 10, 255));
        } else {
            scene_dungeon();
            if (G.state == STATE_DIALOG || G.state == STATE_GENERATING)
                scene_dialog_box();
        }

        // Wait for RDP to finish before CPU text pass
        rdpq_detach_wait();

        // ── CPU text pass (same surface, no buffer switch → no flicker) ───
        draw_text(disp);

        display_show(disp);

        // ── Music ──────────────────────────────────────────────────────────
        music_update();

        // ── Input ──────────────────────────────────────────────────────────
        handle_input();
    }

    return 0;
}
