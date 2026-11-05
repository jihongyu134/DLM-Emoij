# -*- coding: utf-8 -*-
"""Chat with the locally trained tiny emoji GPT.

Usage:
  python -I chat_tiny.py                        # auto-detects system prompt by mood
  python -I chat_tiny.py --mood neutral         # plain assistant, no emoji
  python -I chat_tiny.py --mood emotional       # warm companion with emojis
  python -I chat_tiny.py --ckpt models/tiny_emoji_gpt/ckpt.pt
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_tiny_gpt import TinyGPT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join("models", "tiny_emoji_gpt_final", "ckpt.pt"))
    ap.add_argument("--mood", choices=["emotional", "neutral", "auto"], default="auto")
    ap.add_argument("--lang", choices=["zh", "en", "auto"], default="auto")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=120)
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        print("checkpoint not found:", args.ckpt)
        sys.exit(1)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg, itos = ckpt["config"], ckpt["itos"]
    stoi = {c: i for i, c in enumerate(itos)}
    sp = ckpt["specials"]
    model = TinyGPT(len(itos), cfg["block"], cfg["n_layer"], cfg["n_embd"], cfg["n_head"],
                    cfg["dropout"])
    model.load_state_dict(ckpt["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print("loaded %s  params %.1fM  device %s  (输入 /exit 退出)"
          % (args.ckpt, sum(p.numel() for p in model.parameters()) / 1e6, device))

    history = []  # list of user messages

    def respond(user_msg):
        history.append(user_msg)
        # history alternates [u1, a1, u2, a2, ...]; keep recent turns within context
        turns = history[-5:]
        text = sp["sys"] + pick_system(user_msg) + sp["sys_end"]
        if len(turns) % 2 == 1:
            pairs = [(turns[i], turns[i + 1]) for i in range(0, len(turns) - 1, 2)]
            tail = sp["u"] + turns[-1] + sp["u_end"] + sp["a"]
        else:
            pairs = [(turns[i], turns[i + 1]) for i in range(0, len(turns) - 2, 2)]
            tail = sp["a"] + turns[-1] + sp["a_end"]
        for u, a in pairs:
            text += sp["u"] + u + sp["u_end"] + sp["a"] + a + sp["a_end"]
        text += tail
        idx = torch.tensor([[stoi.get(c, 0) for c in text]], device=device)
        with torch.no_grad():
            for _ in range(args.max_new):
                ctx = idx[:, -cfg["block"]:]
                logits, _ = model(ctx)
                logits = logits[:, -1, :] / args.temperature
                v, _ = torch.topk(logits, args.topk)
                logits[logits < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, -1), 1)
                idx = torch.cat([idx, nxt], 1)
                if nxt.item() == stoi[sp["eos"]]:
                    break
        gen = "".join(itos[i] for i in idx[0].tolist()[len(text):])
        reply = gen.split(sp["a_end"])[0].replace(sp["a"], "").strip()
        history.append(reply)
        return reply

    def pick_system(user_msg):
        if args.mood == "neutral":
            return ckpt["sys_prompts"]["zh_neu" if looks_zh(user_msg) else "en_neu"]
        if args.mood == "emotional":
            return ckpt["sys_prompts"]["zh_emo" if looks_zh(user_msg) else "en_emo"]
        # auto: heuristic on emoji/emotion words in the user's message
        emo_words = ["难过", "伤心", "开心", "生气", "害怕", "累", "爱", "笑", "哭", "烦",
                     "sad", "happy", "angry", "scared", "tired", "love", "lol", "great"]
        if any(w in user_msg for w in emo_words) or "😀" in user_msg:
            return ckpt["sys_prompts"]["zh_emo" if looks_zh(user_msg) else "en_emo"]
        return ckpt["sys_prompts"]["zh_neu" if looks_zh(user_msg) else "en_neu"]

    def looks_zh(s):
        return any("\u4e00" <= c <= "\u9fff" for c in s)

    print("mood:", args.mood)
    while True:
        try:
            user = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user.lower() in ("/exit", "/quit", "q"):
            break
        print("AI>", respond(user))
    print("bye!")


if __name__ == "__main__":
    main()
