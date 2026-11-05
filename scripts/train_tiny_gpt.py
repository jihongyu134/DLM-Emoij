# -*- coding: utf-8 -*-
"""Train a tiny (~100M param) char-level GPT locally on the emoji dialogue dataset.

- Python 3.8 / torch 2.x, single GPU (RTX 3050 6GB tested), no numpy needed.
- Data: every raw/*.jsonl dialogue is validated, rendered to a single text
  stream with role markers; loss is computed on assistant spans only.
- Char-level vocab (Chinese + ASCII + emoji) built from the corpus.

Usage:
  python -I train_tiny_gpt.py --raw-dir raw --out-dir models/tiny_emoji_gpt
  python -I train_tiny_gpt.py --raw-dir raw --out-dir models/tiny_emoji_gpt --steps 3000
"""
import argparse
import glob
import json
import math
import os
import random
import re
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_print = print
def print(*a, **k):  # noqa: A001 - stream progress live to the dashboard log
    k["flush"] = True
    _print(*a, **k)

# ----------------------------------------------------------------- config
SYS_ZH_EMO = "你是一个温暖贴心的聊天伙伴，善于倾听、共情和鼓励，回复自然口语化，会在合适的时机使用emoji。"
SYS_ZH_NEU = "你是一个知识丰富、乐于助人的智能助手，回答清晰、准确、有条理。"
SYS_EN_EMO = ("You are a warm, caring chat companion who listens, empathizes, and encourages. "
              "You speak naturally and use emojis at fitting moments.")
SYS_EN_NEU = ("You are a knowledgeable, helpful assistant who gives clear, accurate, "
              "well-organized answers.")
EMOTIONS = {"joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"}
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\u2764\uFE0F\u203C\u2049\u20E3]")

T_SYS, T_SYS_END = "<sys>", "</sys>"
T_U, T_U_END = "<u>", "</u>"
T_A, T_A_END = "<a>", "</a>"
T_EOS = "<eos>"
SPECIALS = [T_SYS, T_SYS_END, T_U, T_U_END, T_A, T_A_END, T_EOS]


def valid_dialogue(obj):
    """Same checks as validate_and_split.py; returns None or the object."""
    if not isinstance(obj, dict):
        return None
    for k in ("system", "messages", "emotion", "lang", "topic"):
        if k not in obj:
            return None
    if obj["emotion"] not in EMOTIONS or obj["lang"] not in ("zh", "en"):
        return None
    msgs = obj["messages"]
    if not isinstance(msgs, list) or len(msgs) < 4 or len(msgs) % 2 != 0:
        return None
    for i, m in enumerate(msgs):
        want = "user" if i % 2 == 0 else "assistant"
        if not isinstance(m, dict) or m.get("role") != want:
            return None
        c = m.get("content", "")
        if not isinstance(c, str) or not c.strip() or len(c) > 600:
            return None
    for m in msgs:
        has = bool(EMOJI_RE.search(m["content"]))
        if obj["emotion"] == "neutral" and has:
            return None
        if obj["emotion"] != "neutral" and m["role"] == "assistant" and not has:
            return None
    return obj


def load_dialogues(raw_dir):
    out, seen = [], set()
    paths = sorted(glob.glob(os.path.join(raw_dir, "**", "*.jsonl"), recursive=True))
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                obj = valid_dialogue(obj)
                if obj is None:
                    continue
                key = (obj["lang"], obj["emotion"],
                       re.sub(r"\s+", " ", obj["messages"][0]["content"].strip().lower()))
                if key in seen:
                    continue
                seen.add(key)
                out.append(obj)
    return out


def render(obj):
    """Dialogue -> (text, assistant_span_flags). flag=1 where loss applies."""
    texts, flags = [], []
    def push(s, learn):
        texts.append(s)
        flags.extend([learn] * len(s))
    push(T_SYS + obj["system"] + T_SYS_END, 0)
    for m in obj["messages"]:
        if m["role"] == "user":
            push(T_U + m["content"] + T_U_END, 0)
        else:
            push(T_A, 0)
            push(m["content"], 1)
            push(T_A_END, 0)
    push(T_EOS, 0)
    return "".join(texts), flags


# ----------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab, block_size, n_layer=12, n_embd=768, n_head=12, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(n_embd, n_head, dropout) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab, bias=False)
        self.head.weight = self.tok.weight  # tie
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos)[None])
        mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), 1)
        for b in self.blocks:
            x = b(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-100)
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="raw")
    ap.add_argument("--out-dir", default="models/tiny_emoji_gpt")
    ap.add_argument("--n-layer", type=int, default=12)
    ap.add_argument("--n-embd", type=int, default=768)
    ap.add_argument("--n-head", type=int, default=12)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-every", type=int, default=500)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    convs = load_dialogues(args.raw_dir)
    if len(convs) < 10:
        print("not enough valid dialogues in", args.raw_dir)
        sys.exit(1)
    print("valid dialogues:", len(convs))

    streams = [render(c) for c in convs]
    corpus = T_EOS.join(s[0] for s in streams) + T_EOS
    print("corpus chars:", len(corpus))

    chars = sorted(set(corpus) - set(SPECIALS))
    itos = list(SPECIALS) + chars
    stoi = {c: i for i, c in enumerate(itos)}
    vocab = len(itos)
    print("vocab:", vocab, "(specials", len(SPECIALS), "+ chars", len(chars), ")")

    ids, labels = [], []
    for text, flags in streams:
        seq = [stoi[T_EOS]] + [stoi[c] for c in text] + [stoi[T_EOS]]
        lab = [-100] + [-100 if f == 0 else stoi[c] for c, f in zip(text, flags)] + [-100]
        ids.extend(seq)
        labels.extend(lab)
    print("total tokens:", len(ids), " learnable:", sum(1 for x in labels if x != -100))

    data = torch.tensor(ids, dtype=torch.long)
    tgt = torch.tensor(labels, dtype=torch.long)
    n_blocks = len(ids) // args.block
    print("blocks of %d: %d -> steps/epoch @batch%d: %d"
          % (args.block, n_blocks, args.batch, max(1, n_blocks // args.batch)))

    model = TinyGPT(vocab, args.block, args.n_layer, args.n_embd, args.n_head, args.dropout).to(device)
    print("params: %.1fM" % (model.num_params() / 1e6))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    best = float("inf")
    for step in range(1, args.steps + 1):
        ix = torch.randint(0, max(1, len(ids) - args.block - 1), (args.batch,))
        x = torch.stack([data[i:i + args.block] for i in ix]).to(device)
        y = torch.stack([tgt[i + 1:i + 1 + args.block] for i in ix]).to(device)
        lr = lr_at(step - 1)
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        vl = loss.item()
        best = min(best, vl)
        if step % 50 == 0 or step == 1:
            spd = (time.time() - t0) / step
            print("step %5d  loss %.4f  best %.4f  lr %.2e  %.2fs/step  eta %.1fmin"
                  % (step, vl, best, lr, spd, spd * (args.steps - step) / 60))
        if step % args.sample_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                prompt = T_SYS + SYS_ZH_EMO + T_SYS_END + T_U + "今天面试通过啦！" + T_U_END + T_A
                ids_p = torch.tensor([[stoi[c] for c in prompt]], device=device)
                out = ids_p
                for _ in range(80):
                    ctx = out[:, -args.block:]
                    logits, _ = model(ctx)
                    logits = logits[:, -1, :] / 0.8
                    v, _ = torch.topk(logits, 40)
                    logits[logits < v[:, [-1]]] = -float("inf")
                    nxt = torch.multinomial(F.softmax(logits, -1), 1)
                    out = torch.cat([out, nxt], 1)
                    if nxt.item() == stoi[T_EOS]:
                        break
                gen = "".join(itos[i] for i in out[0].tolist()[len(ids_p[0]):])
            print("  [sample] 今天面试通过啦！ ->", gen.replace(T_A_END, ""))
            model.train()

    ckpt = {
        "model": model.state_dict(),
        "config": {"n_layer": args.n_layer, "n_embd": args.n_embd, "n_head": args.n_head,
                   "block": args.block, "dropout": 0.0},
        "itos": itos,
        "sys_prompts": {"zh_emo": SYS_ZH_EMO, "zh_neu": SYS_ZH_NEU,
                        "en_emo": SYS_EN_EMO, "en_neu": SYS_EN_NEU},
        "specials": {"sys": T_SYS, "sys_end": T_SYS_END, "u": T_U, "u_end": T_U_END,
                     "a": T_A, "a_end": T_A_END, "eos": T_EOS},
        "best_loss": best,
    }
    path = os.path.join(args.out_dir, "ckpt.pt")
    torch.save(ckpt, path)
    print("saved:", path, " best train loss: %.4f" % best, " total %.1fmin"
          % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
