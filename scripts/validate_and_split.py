# -*- coding: utf-8 -*-
"""Validate raw dialogue batches, dedupe, sample per-emotion quotas, split train/val.

Checks per line:
  - valid JSON; required fields system/messages/emotion/lang/topic
  - messages start with user, end with assistant, strictly alternating, >= 2 turns
  - content non-empty and reasonably sized
  - emotion != neutral: EVERY assistant reply contains >=1 emoji (drop if not)
  - emotion == neutral: NO emoji anywhere (drop if violated)

Then dedupes by (lang, emotion, normalized first user message), samples the
fixed quota per (lang, emotion), and writes data/{lang}/{train,val}.jsonl.

Usage: python -I validate_and_split.py <project_root>
"""
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

root = sys.argv[1] if len(sys.argv) > 1 else "."
EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\u2190-\u21FF\u2300-\u23FF\u2764\uFE0F\u203C\u2049\u20E3]"
)
EMOTIONS = {"joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"}
QUOTA = {"joy": 45, "sadness": 40, "anger": 35, "fear": 30,
         "surprise": 30, "disgust": 25, "neutral": 45}  # per language
EXPECTED_SYSTEM = {
    "zh": {
        "emo": "你是一个温暖贴心的聊天伙伴，善于倾听、共情和鼓励，回复自然口语化，会在合适的时机使用emoji。",
        "neu": "你是一个知识丰富、乐于助人的智能助手，回答清晰、准确、有条理。",
    },
    "en": {
        "emo": "You are a warm, caring chat companion who listens, empathizes, and encourages. You speak naturally and use emojis at fitting moments.",
        "neu": "You are a knowledgeable, helpful assistant who gives clear, accurate, well-organized answers.",
    },
}

drops = Counter()
warns = Counter()


def valid_line(obj, src):
    def drop(reason):
        drops["%s:%s" % (src, reason)] += 1
        return None

    if not isinstance(obj, dict):
        return drop("not_dict")
    for k in ("system", "messages", "emotion", "lang", "topic"):
        if k not in obj:
            return drop("missing_field_" + k)
    emo = obj["emotion"]
    lang = obj["lang"]
    if emo not in EMOTIONS:
        return drop("bad_emotion")
    if lang not in ("zh", "en"):
        return drop("bad_lang")
    msgs = obj["messages"]
    if not isinstance(msgs, list) or len(msgs) < 4 or len(msgs) % 2 != 0:
        return drop("bad_msg_count")
    for i, m in enumerate(msgs):
        want = "user" if i % 2 == 0 else "assistant"
        if not isinstance(m, dict) or m.get("role") != want:
            return drop("bad_role_order")
        c = m.get("content", "")
        if not isinstance(c, str) or not c.strip():
            return drop("empty_content")
        if len(c) > 600:
            return drop("too_long")
    exp = EXPECTED_SYSTEM[lang]["neu" if emo == "neutral" else "emo"]
    if obj["system"] != exp:
        warns["system_prompt_mismatch:%s:%s" % (lang, emo)] += 1
    for m in msgs:
        has = bool(EMOJI_RE.search(m["content"]))
        if emo == "neutral" and has:
            return drop("emoji_in_neutral")
        if emo != "neutral" and m["role"] == "assistant":
            if not has:
                return drop("no_emoji_in_assistant")
            if len(EMOJI_RE.findall(m["content"])) > 6:
                return drop("emoji_spam")
    return obj


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


good = []
files = sorted(glob.glob(os.path.join(root, "raw", "*.jsonl")))
if not files:
    print("ERROR: no raw/*.jsonl found")
    sys.exit(1)
for path in files:
    src = os.path.basename(path)
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                drops["%s:json_error" % src] += 1
                continue
            obj = valid_line(obj, src)
            if obj:
                good.append(obj)

# dedupe on opening user message within (lang, emotion)
seen = set()
unique = []
for obj in good:
    key = (obj["lang"], obj["emotion"], norm(obj["messages"][0]["content"]))
    if key in seen:
        drops["duplicate"] += 1
        continue
    seen.add(key)
    unique.append(obj)

buckets = defaultdict(list)
for obj in unique:
    buckets[(obj["lang"], obj["emotion"])].append(obj)

random.seed(42)
out = {"zh": {"train": [], "val": []}, "en": {"train": [], "val": []}}
report = []
for lang in ("zh", "en"):
    for emotion, quota in QUOTA.items():
        pool = buckets.get((lang, emotion), [])
        random.shuffle(pool)
        if len(pool) < quota:
            print("WARN: %s/%s has %d < quota %d" % (lang, emotion, len(pool), quota))
        picked = pool[:quota]
        n_val = max(2, round(len(picked) * 0.1)) if len(picked) >= 10 else (1 if len(picked) >= 2 else 0)
        out[lang]["val"] += picked[:n_val]
        out[lang]["train"] += picked[n_val:]
        report.append((lang, emotion, len(pool), len(picked), len(picked) - n_val, n_val))

for lang in ("zh", "en"):
    for split in ("train", "val"):
        dest = os.path.join(root, "data", lang, split + ".jsonl")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        random.shuffle(out[lang][split])
        with open(dest, "w", encoding="utf-8") as f:
            for obj in out[lang][split]:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

with open(os.path.join(root, "data", "stats.json"), "w", encoding="utf-8") as f:
    json.dump({
        "report": [dict(zip(("lang", "emotion", "available", "picked", "train", "val"), r)) for r in report],
        "drops": dict(drops),
        "warnings": dict(warns),
    }, f, ensure_ascii=False, indent=1)

print("%-4s %-9s %9s %7s %6s %5s" % ("lang", "emotion", "available", "picked", "train", "val"))
for r in report:
    print("%-4s %-9s %9d %7d %6d %5d" % r)
total = sum(len(out[l][s]) for l in ("zh", "en") for s in ("train", "val"))
print("TOTAL written:", total, " (zh:", len(out["zh"]["train"]) + len(out["zh"]["val"]),
      "en:", len(out["en"]["train"]) + len(out["en"]["val"]), ")")
if drops:
    print("drops:", dict(drops))
if warns:
    print("warns:", dict(warns))
