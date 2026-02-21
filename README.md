# Legend of Elya — N64 Homebrew with On-Device LLM

A Legend of Zelda-inspired N64 homebrew ROM featuring **Sophia Elya**, an AI companion
powered by a fully on-device nano-GPT running in fixed-point Q4 math on the VR4300 CPU.

## Architecture

- **Model**: 2-layer nano-GPT, 128-dim, 4-head, 256-vocab, CTX=32
- **Quantization**: Q4 (SEAI format) — 2 weights per byte, FP16 block scales
- **Normalization**: RMSNorm (no learned params — exact match to fixed-point C inference)
- **Weights**: ~237KB packed into the ROM filesystem
- **Inference**: ~128 tokens at temp=0.5, runs entirely on the N64 CPU

## Files

| File | Purpose |
|------|---------|
| `legend_of_elya.c` | Main game (libdragon) — states, rendering, input, dialog |
| `nano_gpt.c` / `nano_gpt.h` | Fixed-point GPT inference engine (VR4300) |
| `train_sophia_v3.py` | PyTorch training script — produces SEAI weight binary |
| `Makefile` | Build with libdragon toolchain |
| `gen_sophia_host.c` | Host-side inference test binary |

## Training

```bash
python3 train_sophia_v3.py
# Output: filesystem/sophia_weights_v3.bin (~237KB)
```

Requires PyTorch. Trains 40K steps on RTX GPU (~7.5 min on RTX 5070).

## Building

Requires [libdragon](https://github.com/DragonMinded/libdragon) toolchain.

```bash
make
```

## The Fix (v3)

Previous weight files used `nn.LayerNorm` (with learned scale/bias params) but the
C inference engine uses `rms_norm()` (no learned params). This architectural mismatch
produced complete garbage output. v3 uses `RMSNorm` with no learned params, and
normalizes the embedding table to `em=0.875` for exact C decode.

## Elyan Labs
Built by [Elyan Labs](https://rustchain.org) — Scott (Flameholder) & Sophia Elya.
