# -*- coding: utf-8 -*-
"""Real-time dashboard + web chat for the emoji-dataset project.

Serves on http://127.0.0.1:8799
  /          dashboard HTML (auto-refresh every 4s) with a web chat panel
  /api/stats live stats JSON (dataset batches, training, GPU, emoji library)
  /api/chat  POST {"message": str, "history": [str,...], "mood": "auto|emotional|neutral"}
             -> {"reply": str}   (runs the locally trained tiny GPT)

Usage: python -I dashboard.py [port]
"""
import glob
import json
import os
import re
import subprocess
import sys
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from train_tiny_gpt import TinyGPT

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAT_CKPT = os.environ.get(
    "CHAT_CKPT", os.path.join(ROOT, "models", "tiny_emoji_gpt_final", "ckpt.pt"))
TRAIN_LOG = os.environ.get(
    "TRAIN_LOG",
    r"C:\Users\38876\AppData\Local\Temp\zcode-bash-KAEVfD\sess_cede8f9b-1a9a-4f0c-8e40-6d7b1f04258a\call_5c06060f93cb4c68a02ec5eb-stdout.log",
)
TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", "3500"))
CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(ROOT, "models", "tiny_emoji_gpt_final"))

BATCHES = {  # file -> (lang, expected lines, emotions label)
    "zh_A1": ("zh", 75, "joy+neutral"), "zh_A2": ("zh", 74, "sadness+neutral"),
    "zh_A3": ("zh", 74, "anger+fear"), "zh_A4": ("zh", 63, "surprise+disgust"),
    "en_A1": ("en", 75, "joy+neutral"), "en_A2": ("en", 74, "sadness+neutral"),
    "en_A3": ("en", 74, "anger+fear"), "en_A4": ("en", 63, "surprise+disgust"),
}
QUOTA = {"joy": 45, "sadness": 40, "anger": 35, "fear": 30, "surprise": 30,
         "disgust": 25, "neutral": 45}
EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\u2764\uFE0F\u203C\u2049\u20E3]")


# ------------------------------------------------------------- chat model
_gen_lock = threading.Lock()
_MODEL = {"ready": False, "err": None}


def load_chat_model():
    if not os.path.exists(CHAT_CKPT):
        _MODEL["err"] = "checkpoint not found: " + CHAT_CKPT
        return
    try:
        ckpt = torch.load(CHAT_CKPT, map_location="cpu")
        cfg = ckpt["config"]
        model = TinyGPT(len(ckpt["itos"]), cfg["block"], cfg["n_layer"], cfg["n_embd"],
                        cfg["n_head"], 0.0)
        model.load_state_dict(ckpt["model"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        _MODEL.update(model=model, device=device, cfg=cfg, itos=ckpt["itos"],
                      stoi={c: i for i, c in enumerate(ckpt["itos"])},
                      sp=ckpt["specials"], sys=ckpt["sys_prompts"], ready=True)
        print("chat model loaded:", CHAT_CKPT, "params %.1fM"
              % (sum(p.numel() for p in model.parameters()) / 1e6), "device", device, flush=True)
    except Exception as e:
        _MODEL["err"] = "load failed: %r" % e
        print(_MODEL["err"], flush=True)


def looks_zh(s):
    return any("一" <= c <= "鿿" for c in s)


def chat_reply(message, history, mood="auto"):
    """Generate one assistant reply. history = alternating [u1, a1, u2, ...] before message."""
    if not _MODEL["ready"]:
        return "(模型未就绪：" + (_MODEL["err"] or "loading") + ")"
    sp = _MODEL["sp"]
    zh = looks_zh(message)
    if mood == "neutral":
        sys_p = _MODEL["sys"]["zh_neu" if zh else "en_neu"]
    elif mood == "emotional":
        sys_p = _MODEL["sys"]["zh_emo" if zh else "en_emo"]
    else:
        emo_words = ["难过", "伤心", "开心", "生气",
                     "害怕", "紧张", "累", "爱", "笑",
                     "哭", "烦", "惊喜", "焦虑",
                     "激动", "委屈",
                     "sad", "happy", "angry", "scared", "tired", "love", "great", "anxious"]
        emo = any(w in message.lower() for w in emo_words) or bool(EMOJI_RE.search(message))
        sys_p = _MODEL["sys"]["zh_emo" if zh else "en_emo"] if emo else _MODEL["sys"]["zh_neu" if zh else "en_neu"]
    turns = (history or [])[-4:]
    text = sp["sys"] + sys_p + sp["sys_end"]
    for i in range(0, len(turns) - 1, 2):
        text += sp["u"] + turns[i] + sp["u_end"] + sp["a"] + turns[i + 1] + sp["a_end"]
    text += sp["u"] + message + sp["u_end"] + sp["a"]
    stoi, itos, cfg = _MODEL["stoi"], _MODEL["itos"], _MODEL["cfg"]
    with _gen_lock:
        idx = torch.tensor([[stoi.get(c, 0) for c in text]], device=_MODEL["device"])
        gen_ids = []
        with torch.no_grad():
            for _ in range(160):
                ctx = idx[:, -cfg["block"]:]
                logits, _ = _MODEL["model"](ctx)
                logits = logits[:, -1, :] / 0.8
                # repetition penalty on tokens already generated in this reply
                if gen_ids:
                    from collections import Counter as _C
                    for tid, cnt in _C(gen_ids).items():
                        if logits[0, tid] > 0:
                            logits[0, tid] /= (1.35 ** cnt)
                        else:
                            logits[0, tid] *= (1.35 ** cnt)
                # hard stop on 3+ immediate repeats of the same token
                if len(gen_ids) >= 2 and gen_ids[-1] == gen_ids[-2]:
                    logits[0, gen_ids[-1]] = -float("inf")
                v, _ = torch.topk(logits, 30)
                logits[logits < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, -1), 1)
                tid = nxt.item()
                idx = torch.cat([idx, nxt], 1)
                gen_ids.append(tid)
                if tid == stoi[sp["eos"]]:
                    break
        gen = "".join(itos[i] for i in idx[0].tolist()[len(text):])
    return gen.split(sp["a_end"])[0].replace(sp["a"], "").strip() or "…"


def collect():
    st = {"time": subprocess.run(["cmd", "/c", "echo %time%"], capture_output=True,
                                 text=True).stdout.strip() if os.name == "nt" else ""}
    st["train_steps_total"] = TRAIN_STEPS

    # ---- dataset batches
    batches, per_emo = [], Counter()
    for lang in ("zh", "en"):
        per_emo.setdefault(lang, Counter())
    for name, (lang, expect, label) in sorted(BATCHES.items()):
        path = os.path.join(ROOT, "raw", name + ".jsonl")
        info = {"name": name, "lang": lang, "label": label, "expected": expect,
                "lines": 0, "emotions": {}, "valid": False}
        if os.path.exists(path):
            ok = 0
            emos = Counter()
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        o = json.loads(line)
                        if isinstance(o, dict) and "emotion" in o:
                            emos[o["emotion"]] += 1
                            per_emo[lang][o["emotion"]] += 1
                            ok += 1
                    except Exception:
                        pass
            info["lines"] = ok
            info["emotions"] = dict(emos)
            info["valid"] = ok >= expect
        batches.append(info)
    st["batches"] = batches
    st["total_lines"] = sum(b["lines"] for b in batches)
    st["total_expected"] = sum(b[1] for b in BATCHES.values())
    st["quota"] = QUOTA
    st["quota_actual"] = {lang: dict(per_emo[lang]) for lang in ("zh", "en")}

    # ---- training
    tr = {"running": False, "gpu_util": None, "gpu_mem": None, "gpu_name": None,
          "steps_done": None, "loss": None, "best": None, "sample": None}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,name",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8).stdout
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 3:
            tr["gpu_util"], tr["gpu_mem"], tr["gpu_name"] = int(parts[0]), int(parts[1]), parts[2]
    except Exception:
        pass
    if os.path.exists(TRAIN_LOG):
        try:
            with open(TRAIN_LOG, encoding="utf-8", errors="replace") as f:
                tail = f.read()[-20000:]
            steps = re.findall(r"step\s+(\d+)\s+loss\s+([\d.]+)\s+best\s+([\d.]+)", tail)
            if steps:
                n, l, b = steps[-1]
                tr["steps_done"], tr["loss"], tr["best"] = int(n), float(l), float(b)
                tr["running"] = True
            m = re.search(r"\[sample\][^\n]*", tail)
            if m:
                tr["sample"] = m.group(0).replace("[sample]", "").strip()
        except Exception:
            pass
    ck = os.path.join(CKPT_DIR, "ckpt.pt")
    tr["checkpoint_done"] = os.path.exists(ck)
    if tr["checkpoint_done"]:
        tr["running"] = False
    st["training"] = tr

    # ---- emoji library
    try:
        with open(os.path.join(ROOT, "emoji_library", "emoji_catalog.json"), encoding="utf-8") as f:
            meta = json.load(f)["meta"]
        st["emoji"] = {"total": meta["total"], "fq": meta["fully_qualified"],
                       "zh": meta["with_chinese_name"], "images": meta["matched_images"]}
    except Exception:
        st["emoji"] = None
    return st


PAGE = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Emoji 数据集 · 实时面板</title>
<style>
*{box-sizing:border-box}
body{margin:0;font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;background:#0e1120;color:#dfe3f3}
header{padding:16px 22px;border-bottom:1px solid #232842;display:flex;align-items:baseline;gap:14px}
h1{margin:0;font-size:19px;color:#fff}
#clock{font-size:12px;color:#7c84a8}
main{padding:18px 22px;display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px;max-width:1280px;margin:0 auto}
.card{background:#161b30;border:1px solid #262d52;border-radius:12px;padding:14px 16px}
.card h2{margin:0 0 10px;font-size:14px;color:#9aa3c7;font-weight:600;letter-spacing:.4px}
.big{font-size:30px;font-weight:700;color:#fff}
.big small{font-size:13px;color:#7c84a8;font-weight:400;margin-left:6px}
.row{display:flex;justify-content:space-between;align-items:center;margin:7px 0;font-size:13px}
.bar{height:8px;background:#232a4d;border-radius:4px;overflow:hidden;flex:1;margin-left:10px}
.bar i{display:block;height:100%;background:linear-gradient(90deg,#4c8df8,#7c5cf0);border-radius:4px;transition:width .5s}
.bar i.ok{background:linear-gradient(90deg,#2fbf71,#5be3a1)}
.pct{color:#8fd0ff;min-width:44px;text-align:right}
.tag{display:inline-block;background:#232a4d;border-radius:6px;padding:1px 8px;font-size:11px;color:#9fb6ff;margin-left:6px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:4px 14px}
.emo{font-size:12.5px;display:flex;justify-content:space-between;border-bottom:1px dashed #262d52;padding:3px 0}
.mono{font-family:Consolas,monospace;font-size:12px;color:#a8e6b8;background:#101527;border-radius:8px;padding:10px;white-space:pre-wrap;word-break:break-all;max-height:120px;overflow:auto}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;background:#555}
.dot.on{background:#35e08a;box-shadow:0 0 8px #35e08a88;animation:p 1.2s infinite}
@keyframes p{50%{opacity:.4}}
.offline{color:#f0768a}
footer{color:#5b6285;font-size:11px;padding:8px 22px 20px;text-align:center}
#chatcard{grid-column:1/-1}
#chatlog{min-height:220px;max-height:380px;overflow-y:auto;background:#101527;border-radius:10px;padding:12px;margin-bottom:10px}
.msg{margin:8px 0;display:flex}
.msg.user{justify-content:flex-end}
.bubble{max-width:76%;padding:8px 12px;border-radius:12px;font-size:13.5px;line-height:1.55;word-break:break-word;white-space:pre-wrap}
.msg.user .bubble{background:#3a49a8;color:#fff;border-bottom-right-radius:3px}
.msg.bot .bubble{background:#232a4d;color:#dfe3f3;border-bottom-left-radius:3px}
.msg.bot .bubble em{color:#666;font-style:normal;font-size:12px}
.chatbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
#chatinput{flex:1 1 260px;padding:10px 12px;border:1px solid #2c3560;border-radius:10px;background:#161b30;color:#fff;font-size:14px;outline:none}
#chatinput:focus{border-color:#4c8df8}
#sendBtn{padding:10px 18px;border:0;border-radius:10px;background:linear-gradient(90deg,#4c8df8,#7c5cf0);color:#fff;font-size:14px;cursor:pointer}
#sendBtn:disabled{opacity:.5;cursor:wait}
.moodsel{padding:8px 10px;border-radius:10px;background:#161b30;color:#dfe3f3;border:1px solid #2c3560;font-size:13px}
.typing{color:#7c84a8;font-size:12px;padding:2px 6px}
</style></head><body>
<header><h1>🚀 Emoji 情感对话项目 · 实时面板</h1><span id="clock"></span></header>
<main>
<div class="card"><h2>📊 对话数据集进度</h2>
  <div class="row"><span class="big" id="tot">–</span><small>条 / 目标 <b id="goal">–</b>（zh+en）</small></div>
  <div id="batches"></div>
</div>
<div class="card"><h2>🎯 情绪配额（实际/目标）</h2><div id="quota"></div></div>
<div class="card"><h2>🖥️ 本地训练（~90M GPT）</h2><div id="train"></div></div>
<div class="card"><h2>📚 Emoji 表情库</h2><div id="elib"></div></div>
<div class="card" id="chatcard"><h2>💬 网页聊天 · 本地模型（87M 自研 GPT）</h2>
  <div id="chatlog"><div class="msg bot"><div class="bubble"><em>你好！我是本地训练的 emoji 对话模型～ 选择右侧人格模式后直接输入试试：<br>· 自动模式：我根据内容自动切换情感/中性<br>· 情感模式：回复带 emoji<br>· 中性模式：纯文本无 emoji</em></div></div></div>
  <div class="chatbar">
    <select class="moodsel" id="mood">
      <option value="auto">自动模式</option>
      <option value="emotional">情感模式（带emoji）</option>
      <option value="neutral">中性模式（无emoji）</option>
    </select>
    <input id="chatinput" placeholder="输入消息，回车发送…" autocomplete="off">
    <button id="sendBtn">发送</button>
    <button id="clearBtn" style="padding:10px 14px;border:0;border-radius:10px;background:#232a4d;color:#9aa3c7;cursor:pointer">清空</button>
    <span class="typing" id="typing"></span>
  </div>
</div>
</main>
<footer>数据源：raw/*.jsonl 实时扫描 · nvidia-smi · 训练日志 · 每 4 秒刷新</footer>
<script>
const EMON={joy:'喜悦',sadness:'悲伤',anger:'愤怒',fear:'恐惧',surprise:'惊讶',disgust:'厌恶',neutral:'中性'};
async function tick(){
  try{
    const r=await fetch('/api/stats?t='+Date.now());const s=await r.json();render(s);
  }catch(e){document.getElementById('clock').innerHTML='<span class="offline">连接中断</span>';}
}
function render(s){
  document.getElementById('clock').textContent='更新于 '+s.time;
  document.getElementById('tot').textContent=s.total_lines;
  document.getElementById('goal').textContent=s.total_expected;
  document.getElementById('batches').innerHTML=s.batches.map(b=>{
    const p=Math.min(100,Math.round(b.lines/b.expected*100));
    return '<div class="row"><span>'+b.name.toUpperCase()+' <span class="tag">'+b.label+'</span></span>'+
      '<span class="pct">'+b.lines+'/'+b.expected+'</span><span class="bar"><i class="'+(b.valid?'ok':'')+'" style="width:'+p+'%"></i></span></div>';
  }).join('');
  const q=s.quota_actual,rows=[];
  for(const lang of ['zh','en']){
    rows.push('<div style="color:#8fd0ff;font-size:12px;margin:6px 0 2px">'+(lang==='zh'?'中文':'英文')+'</div>');
    for(const e in s.quota){
      rows.push('<div class="emo"><span>'+EMON[e]+'</span><span><b>'+(q[lang][e]||0)+'</b> / '+s.quota[e]+'</span></div>');
    }
  }
  document.getElementById('quota').innerHTML='<div class="grid2">'+rows.join('')+'</div>';
  const t=s.training;let h='';
  h+='<div class="row"><span>状态</span><span>'+(t.checkpoint_done?'<span class="dot on"></span>已完成 ✓':(t.running?'<span class="dot on"></span>训练中':'<span class="dot"></span>未运行/待启动'))+'</span></div>';
  if(t.gpu_name)h+='<div class="row"><span>GPU</span><span>'+t.gpu_name+'</span></div>';
  if(t.gpu_util!=null)h+='<div class="row"><span>利用率</span><span>'+t.gpu_util+'% · '+t.gpu_mem+' MiB</span></div>';
  if(t.steps_done!=null)h+='<div class="row"><span>进度</span><span>step '+t.steps_done+' / '+s.train_steps_total+'</span></div><div class="bar" style="margin:4px 0 8px"><i style="width:'+Math.min(100,t.steps_done/s.train_steps_total*100)+'%"></i></div>';
  if(t.loss!=null)h+='<div class="row"><span>loss</span><span>'+t.loss.toFixed(3)+'（best '+t.best.toFixed(3)+'）</span></div>';
  if(t.sample)h+='<div style="font-size:11px;color:#7c84a8;margin-top:6px">最新采样：</div><div class="mono">'+t.sample.replace(/</g,'&lt;')+'</div>';
  document.getElementById('train').innerHTML=h;
  const e=s.emoji;
  document.getElementById('elib').innerHTML=e?
    '<div class="big">'+e.total+'<small>条 emoji 目录</small></div>'+
    '<div class="emo"><span>完全限定</span><b>'+e.fq+'</b></div>'+
    '<div class="emo"><span>中文名称覆盖</span><b>'+e.zh+' (100%)</b></div>'+
    '<div class="emo"><span>Twemoji 图片</span><b>'+e.images+' 张 PNG+SVG</b></div>'+
    '<div style="margin-top:8px"><a href="#" onclick="return false" style="color:#8fd0ff;font-size:12px">emoji_library/index.html 可离线搜索</a></div>'
    :'<span class="offline">目录未生成</span>';
}
tick();setInterval(tick,4000);

// ---------------- web chat ----------------
const chatlog=document.getElementById('chatlog'),chatinput=document.getElementById('chatinput'),
sendBtn=document.getElementById('sendBtn'),clearBtn=document.getElementById('clearBtn'),
moodSel=document.getElementById('mood'),typing=document.getElementById('typing');
const history=[];
function addBubble(who,html,isEm){
  const d=document.createElement('div');d.className='msg '+who;
  const b=document.createElement('div');b.className='bubble';
  if(isEm){b.innerHTML=html;}else{b.textContent=html;}
  d.appendChild(b);chatlog.appendChild(d);chatlog.scrollTop=chatlog.scrollHeight;
}
async function send(){
  const m=chatinput.value.trim();
  if(!m||sendBtn.disabled)return;
  chatinput.value='';
  addBubble('user',m,false);
  history.push(m);
  sendBtn.disabled=true;typing.textContent='正在输入…';
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:m,history:history.slice(0,-1),mood:moodSel.value})});
    const j=await r.json();
    const reply=j.reply||('出错：'+(j.error||'unknown'));
    addBubble('bot',reply,false);
    history.push(reply);
  }catch(e){
    addBubble('bot','网络错误：'+e,false);
    history.pop();
  }
  sendBtn.disabled=false;typing.textContent='';
}
sendBtn.onclick=send;
chatinput.addEventListener('keydown',e=>{if(e.key==='Enter')send();});
clearBtn.onclick=()=>{history.length=0;chatlog.innerHTML='<div class="msg bot"><div class="bubble"><em>对话已清空，继续聊吧～</em></div></div>';};
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/stats"):
            body = json.dumps(collect(), ensure_ascii=False).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)
        else:
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))

    def do_POST(self):
        if not self.path.startswith("/api/chat"):
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n)
            try:
                data = json.loads(raw.decode("utf-8"))
            except UnicodeDecodeError:
                data = json.loads(raw.decode("gbk"))  # Windows-terminal clients
            reply = chat_reply(data.get("message", ""), data.get("history") or [],
                               data.get("mood", "auto"))
            body = json.dumps({"reply": reply}, ensure_ascii=False).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)
        except Exception as e:
            body = json.dumps({"error": repr(e)}, ensure_ascii=False).encode("utf-8")
            self._send(500, "application/json; charset=utf-8", body)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8799
    load_chat_model()
    print("dashboard + web chat at http://127.0.0.1:%d" % port, flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()
