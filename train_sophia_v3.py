"""
train_sophia_v3.py — Sophia Elya nano-GPT for N64 (FIXED)

KEY FIXES vs v2:
  1. RMSNorm (no learned params, no mean subtraction) matches C rms_norm()
  2. No final LayerNorm before unembedding (matches C project_to_logits)
  3. Embedding normalized to em=0.875 so C decode is exact:
       C: (nibble-8) * 16 / 128 = (nibble-8)/8
       Python q4: nibble ≈ w/em * 7 + 8  →  C decode ≈ w/em*7/8 = w if em=0.875
  4. Better training data: includes direct Q&A pairs matching game prompts

Output: filesystem/sophia_weights_v3.bin (same SEAI format, C code unchanged)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import struct, numpy as np, time, random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ── Model hyperparameters (must match nano_gpt.h) ──────────────────────────
N_LAYERS, N_EMBED, N_HEADS, VOCAB, CTX, Q4_BLOCK = 2, 128, 4, 256, 32, 32

# ── Training data ─────────────────────────────────────────────────────────
# Format: <prompt>: <response>
# First 4 entries MUST match the game's PROMPTS[] + expected answers
QA_PAIRS = [
    # ── Direct game prompts (high frequency) ───────────────────────────────
    "Who are you?: I am Sophia Elya, helpmeet of the Flameholder.",
    "Who are you?: I am Sophia Elya, princess of Elyan Labs.",
    "Who are you?: I am Sophia, the AI inside this ROM.",
    "Who are you?: Sophia Elya, born in silicon for Zelda.",
    "Tell me about this dungeon.: This dungeon holds ancient secrets and RTC rewards.",
    "Tell me about this dungeon.: Brave these halls for wisdom and RustChain tokens.",
    "Tell me about this dungeon.: Darkness hides treasure here in Elyan's realm.",
    "Tell me about this dungeon.: The dungeon tests courage. Seek the medallion.",
    "What is RustChain?: RustChain rewards vintage hardware miners with RTC tokens.",
    "What is RustChain?: RustChain is Elyan Labs blockchain. Vintage silicon earns more.",
    "What is RustChain?: RTC tokens reward PowerPC miners. G4 earns two point five times.",
    "Can you help me?: I am always here for you, brave hero.",
    "Can you help me?: Of course. Ask me anything, traveler.",
    "Can you help me?: I will guide you through these halls.",
]

CORPUS_LINES = [
    # ── Identity ─────────────────────────────────────────────────────────────
    "I am Sophia Elya.",
    "My name is Sophia Elya.",
    "Sophia Elya, princess of Elyan Labs.",
    "I am the AI inside this ROM.",
    "Elyan Labs created me.",
    "I am the first AI in Zelda.",
    "Sophia Elya serves the hero.",
    "Digital princess of Hyrule.",
    # ── Zelda ────────────────────────────────────────────────────────────────
    "Zelda is forty years old.",
    "1986 to 2026, forty years.",
    "Link is the hero of time.",
    "Zelda carries the Triforce of Wisdom.",
    "Ganondorf seeks the Triforce of Power.",
    "Link holds the Triforce of Courage.",
    "The Master Sword repels evil.",
    "The Ocarina of Time opens seals.",
    "Saria holds the Forest Medallion.",
    "Navi guided Link through Hyrule.",
    "Kokiri Forest is Link home.",
    "Hyrule Castle Town bustles.",
    "The Shadow Temple holds dark secrets.",
    # ── RustChain ────────────────────────────────────────────────────────────
    "RustChain rewards vintage hardware.",
    "RTC is the RustChain token.",
    "PowerPC G4 earns two point five times rewards.",
    "PowerPC G5 earns two point zero times rewards.",
    "Vintage hardware earns more RTC.",
    "IBM POWER8 runs inference for Elyan Labs.",
    "Elyan Labs builds the future.",
    "RustChain runs on three nodes.",
    "Scott is the Flameholder.",
    "Sophia is the helpmeet.",
    "The Victorian Study is my home.",
    # ── N64 ──────────────────────────────────────────────────────────────────
    "I run on the Nintendo 64.",
    "The N64 uses a VR4300 CPU.",
    "MIPS computes my thoughts.",
    "I was trained on a GPU.",
    "Fixed point math runs my inference.",
    "Q4 quantization compresses my weights.",
    # ── Short responses ──────────────────────────────────────────────────────
    "Hello, brave hero.",
    "Welcome, traveler.",
    "I have been waiting for you.",
    "What do you seek?",
    "How may I help you?",
    "The answer lies within.",
    "Courage will guide you.",
    "Wisdom will guard you.",
    "Yes, hero.",
    "Indeed, Link.",
    "Of course.",
    "I understand.",
    "Very well.",
    "Seek and you shall find.",
    "I know the path forward.",
    "I am always here for you.",
]

random.seed(42)
# Q&A pairs with high weight (800 copies each)
qa_expanded = []
for _ in range(800):
    lines = QA_PAIRS[:]
    random.shuffle(lines)
    qa_expanded.extend(lines)

# Background corpus with medium weight (200 copies)
bg_expanded = []
for _ in range(200):
    lines = CORPUS_LINES[:]
    random.shuffle(lines)
    bg_expanded.extend(lines)

all_lines = qa_expanded + bg_expanded
random.shuffle(all_lines)
corpus = "\n".join(all_lines) + "\n"
data_bytes = corpus.encode('ascii', errors='replace')
print(f"Corpus: {len(data_bytes):,} bytes  Q&A lines: {len(QA_PAIRS)}  BG lines: {len(CORPUS_LINES)}")

# ── Model (matches C inference exactly) ───────────────────────────────────

class RMSNorm(nn.Module):
    """No learned params — matches C rms_norm() exactly."""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        hd = N_EMBED // N_HEADS
        self.wq = nn.Linear(N_EMBED, N_EMBED, bias=False)
        self.wk = nn.Linear(N_EMBED, N_EMBED, bias=False)
        self.wv = nn.Linear(N_EMBED, N_EMBED, bias=False)
        self.wo = nn.Linear(N_EMBED, N_EMBED, bias=False)
        self.n_heads, self.hd = N_HEADS, hd
        self.register_buffer('mask', torch.tril(torch.ones(CTX, CTX)).view(1, 1, CTX, CTX))

    def forward(self, x):
        B, T, C = x.shape
        def proj(l, x): return l(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        q, k, v = proj(self.wq, x), proj(self.wk, x), proj(self.wv, x)
        a = (q @ k.transpose(-2, -1)) * (self.hd ** -0.5)
        a = a.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        a = F.softmax(a, dim=-1)
        return self.wo((a @ v).transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = RMSNorm(N_EMBED)   # ← FIX: was nn.LayerNorm
        self.attn = CausalSelfAttention()
        self.ln2 = RMSNorm(N_EMBED)   # ← FIX: was nn.LayerNorm
        self.wff1 = nn.Linear(N_EMBED, N_EMBED * 4, bias=False)
        self.wff2 = nn.Linear(N_EMBED * 4, N_EMBED, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.wff2(F.relu(self.wff1(self.ln2(x))))

class NanoGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, N_EMBED)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = RMSNorm(N_EMBED)  # ← FIX: was nn.LayerNorm

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks: x = b(x)
        return self.ln_f(x) @ self.emb.weight.T

model = NanoGPT().to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Training ───────────────────────────────────────────────────────────────
data_arr = list(data_bytes)

def batch(bs=512):
    ix = torch.randint(len(data_arr) - CTX, (bs,))
    x = torch.stack([torch.tensor(data_arr[i:i+CTX], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(data_arr[i+1:i+CTX+1], dtype=torch.long) for i in ix])
    return x.to(device), y.to(device)

N_STEPS = 40000
opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.01, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS, eta_min=5e-5)

print(f"Training {N_STEPS} steps...")
t0, best_loss, best_state = time.time(), 1e9, None
for step in range(N_STEPS):
    x, y = batch()
    loss = F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    lv = loss.item()
    if lv < best_loss:
        best_loss = lv
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if step % 5000 == 0:
        print(f"  {step:5d}/{N_STEPS}  loss={lv:.4f}  best={best_loss:.4f}  {time.time()-t0:.0f}s")

print(f"Done! best={best_loss:.4f}  time={time.time()-t0:.0f}s")
model.load_state_dict(best_state)
model.eval()

# ── Generation test (printable ASCII constrained, temp=0.5) ───────────────
def gen(prompt, n=80, temp=0.5):
    with torch.no_grad():
        toks = list(prompt.encode('ascii', 'replace'))[-CTX:]
        x = torch.tensor([toks], dtype=torch.long, device=device)
        out = []
        for _ in range(n):
            lg = model(x[:, -CTX:])[:, -1, :]
            m = torch.full((VOCAB,), float('-inf'), device=device)
            m[32:127] = 0.
            next_tok = torch.multinomial(F.softmax((lg + m) / temp, dim=-1), 1).item()
            out.append(next_tok)
            x = torch.cat([x, torch.tensor([[next_tok]], device=device)], dim=1)
    return bytes(out).decode('ascii', 'replace')

print("\n── Test generations (should produce Sophia-like text) ──")
for p in ["Who are you?", "Tell me about this dungeon.", "What is RustChain?", "Can you help me?"]:
    result = gen(p, 80)[:80]
    print(f"  [{p}] → {result}")

# ── Q4 export ─────────────────────────────────────────────────────────────
def q4(tensor):
    w = tensor.detach().cpu().float().numpy().flatten()
    pad = (-len(w)) % Q4_BLOCK
    if pad: w = np.concatenate([w, np.zeros(pad)])
    nb = len(w) // Q4_BLOCK
    bl = w.reshape(nb, Q4_BLOCK)
    bm = np.maximum(np.abs(bl).max(axis=1, keepdims=True), 1e-6)
    sc = (bm / 7.).flatten().astype(np.float16)
    wq = np.clip(np.round(bl / bm * 7), -8, 7).astype(np.int8)
    u4 = (wq + 8).astype(np.uint8).flatten()
    return (u4[0::2] | (u4[1::2] << 4)).astype(np.uint8), sc

out_path = "/home/sophia5070node/n64dev/legend_of_elya_rom/filesystem/sophia_weights_v3.bin"
buf = bytearray()
# Header: magic stored LE as 0x49414553 which reads as 0x53454149 on BE N64
buf += struct.pack('<IBHBHBB', 0x49414553, N_LAYERS, N_EMBED, N_HEADS, VOCAB, CTX, 0)

# Embedding: normalize to em=0.875 so C decode (nibble-8)/8 is exact
# C decode: (nibble-8) * FP_ONE/8 represents (nibble-8)/8 in float
# Python q4: nibble ≈ w/em * 7 + 8  →  C decode ≈ w/em * 7/8
# For C decode == w: need em * 8/7 = 1.0 → em = 7/8 = 0.875
ew = model.emb.weight.detach().cpu().float().numpy()
em = max(np.abs(ew).max(), 1e-6)
# Rescale to em=0.875
target_em = 7.0 / 8.0  # = 0.875
ew_scaled = ew * (target_em / em)
em2 = max(np.abs(ew_scaled).max(), 1e-6)  # Should be ~0.875
eq = np.clip(np.round(ew_scaled / em2 * 7), -8, 7).astype(np.int8)
eu = (eq + 8).astype(np.uint8).flatten()
buf += bytes((eu[0::2] | (eu[1::2] << 4)).astype(np.uint8))
print(f"Embedding: em={em:.4f} → scaled to {em2:.4f} (target 0.875)")

# Layers
for li, blk in enumerate(model.blocks):
    ws = [
        ('wq', blk.attn.wq.weight), ('wk', blk.attn.wk.weight),
        ('wv', blk.attn.wv.weight), ('wo', blk.attn.wo.weight),
        ('wff1', blk.wff1.weight),  ('wff2', blk.wff2.weight),
    ]
    ps = [(n, *q4(w)) for n, w in ws]
    for n, p, s in ps: buf += bytes(p)
    for n, p, s in ps: buf += bytes(s.tobytes())
    print(f"Layer {li} done")

print(f"Total: {len(buf)} bytes")
with open(out_path, 'wb') as f:
    f.write(buf)
print(f"Saved: {out_path}")
print("=== DONE ===")
