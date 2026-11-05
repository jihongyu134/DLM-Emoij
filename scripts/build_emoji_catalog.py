# -*- coding: utf-8 -*-
"""Build the complete emoji catalog.

Inputs  (under <base>):
  src/emoji-test.txt        Unicode emoji-test.txt (16.0)
  src/annotations_en.xml    CLDR English annotations
  src/annotations_zh.xml    CLDR Chinese (Simplified) annotations
  images/png72/*.png        Twemoji v17.0.3 72x72 PNGs
  images/svg/*.svg          Twemoji v17.0.3 SVGs

Outputs (under <base>):
  emoji_catalog.json        full structured catalog
  emoji_catalog.csv         Excel-friendly CSV (utf-8-sig)
  index.html                offline searchable browser

Usage: python -I build_emoji_catalog.py <emoji_library_dir>
"""
import csv
import datetime
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

base = sys.argv[1] if len(sys.argv) > 1 else "."

GROUP_ZH = {
    "Smileys & Emotion": "笑脸与情感",
    "People & Body": "人与身体",
    "Animals & Nature": "动物与自然",
    "Food & Drink": "食物与饮料",
    "Travel & Places": "旅行与地点",
    "Activities": "活动",
    "Objects": "物品",
    "Symbols": "符号",
    "Flags": "旗帜",
    "Component": "组件",
}

# ---------------------------------------------------------------- emoji-test
entries = []
group = subgroup = ""
test_path = os.path.join(base, "src", "emoji-test.txt")
with open(test_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("# group:"):
            group = line.split(":", 1)[1].strip()
            continue
        if line.startswith("# subgroup:"):
            subgroup = line.split(":", 1)[1].strip()
            continue
        if not line or line.startswith("#"):
            continue
        left, _, right = line.partition(";")
        status = right.split("#", 1)[0].strip()
        after = right.split("#", 1)[1] if "#" in right else ""
        m = re.match(r"\s*(\S+)\s+E(\d+\.\d+)\s+(.+)$", after)
        if not m:
            continue
        emoji, ever, name = m.group(1), m.group(2), m.group(3).strip()
        codes = [c.strip().lower() for c in left.split()]
        entries.append(
            {
                "emoji": emoji,
                "codes": codes,
                "group": group,
                "subgroup": subgroup,
                "status": status,
                "emoji_version": "E" + ever,
                "name_en_test": name,
            }
        )

# ------------------------------------------------------------- CLDR names
def parse_xml_ann(path):
    tts, kw = {}, {}
    if not os.path.exists(path):
        return tts, kw
    root = ET.parse(path).getroot()
    for a in root.iter("annotation"):
        cp = a.get("cp")
        if not cp:
            continue
        text = (a.text or "").strip()
        if a.get("type") == "tts":
            tts[cp] = text
        elif text:
            kw.setdefault(cp, [])
            kw[cp] += [p.strip() for p in text.split("|") if p.strip()]
    return tts, kw

def parse_json_ann(path):
    tts, kw = {}, {}
    if not os.path.exists(path):
        return tts, kw
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    section = None
    for top in ("annotations", "annotationsDerived"):
        if top in d and isinstance(d[top], dict):
            section = d[top].get("annotations")
            break
    if not isinstance(section, dict):
        return tts, kw
    for cp, info in section.items():
        if not isinstance(info, dict):
            continue
        t = info.get("tts") or []
        if t:
            tts[cp] = t[0]
        kw[cp] = list(info.get("default") or [])
    return tts, kw

def merged_names(loc):
    """main tables first, derived tables fill the gaps (skin tones, flags, ZWJ)."""
    sources = [
        (os.path.join(base, "src", "json_%s_main.json" % loc), "json"),
        (os.path.join(base, "src", "annotations_%s.xml" % loc), "xml"),
        (os.path.join(base, "src", "json_%s_derived.json" % loc), "json"),
        (os.path.join(base, "src", "derived_%s.xml" % loc), "xml"),
    ]
    tts, kw = {}, {}
    for path, kind in sources:
        t, k = parse_json_ann(path) if kind == "json" else parse_xml_ann(path)
        for cp, v in t.items():
            tts.setdefault(cp, v)
        for cp, vals in k.items():
            merged = kw.setdefault(cp, [])
            for x in vals:
                if x not in merged:
                    merged.append(x)
            kw[cp] = merged[:12]
    # CLDR keys and emoji-test chars may differ by U+FE0F; alias both spellings
    for table in (tts, kw):
        for cp in list(table.keys()):
            stripped = "".join(ch for ch in cp if ch != "\ufe0f")
            if stripped != cp and stripped not in table:
                table[stripped] = table[cp]
    return tts, kw

en_tts, en_kw = merged_names("en")
zh_tts, zh_kw = merged_names("zh")

# ---------------------------------------------------------------- images
def load_stems(sub):
    d = os.path.join(base, "images", sub)
    return {os.path.splitext(x)[0] for x in os.listdir(d) if not x.startswith(".")}

png_stems = load_stems("png72")
svg_stems = load_stems("svg")

def key_variants(codes):
    """Twemoji filenames are lowercase hex joined by '-', sometimes zero-padded."""
    stripped = [c.lstrip("0") or "0" for c in codes]
    padded = [format(int(c, 16), "04x") for c in codes]
    joins = []
    for parts in (codes, stripped, padded):
        joins.append("-".join(parts))
        nofe0f = [p for p in parts if p != "fe0f"]
        if len(nofe0f) != len(parts) and nofe0f:
            joins.append("-".join(nofe0f))
    if len(codes) == 1:
        joins.append(stripped[0] + "-fe0f")
        joins.append(padded[0] + "-fe0f")
    seen, out = set(), []
    for j in joins:
        if j and j not in seen:
            seen.add(j)
            out.append(j)
    return out

def match_image(codes, stems):
    for v in key_variants(codes):
        if v in stems:
            return v
    return None

# ---------------------------------------------------------------- records
def strip_fe0f(s):
    return "".join(ch for ch in s if ch != "\ufe0f")

records, missing_img = [], 0
for e in entries:
    codes = e["codes"]
    png = match_image(codes, png_stems)
    svg = match_image(codes, svg_stems)
    if png is None:
        missing_img += 1
    emo = e["emoji"]
    emo_ns = strip_fe0f(emo)
    name_en = en_tts.get(emo) or en_tts.get(emo_ns) or e["name_en_test"]
    name_zh = zh_tts.get(emo) or zh_tts.get(emo_ns) or ""
    kwe = en_kw.get(emo) or en_kw.get(emo_ns) or []
    kwz = zh_kw.get(emo) or zh_kw.get(emo_ns) or []
    rec = {
        "emoji": e["emoji"],
        "codes": "-".join(codes),
        "group": e["group"],
        "group_zh": GROUP_ZH.get(e["group"], e["group"]),
        "subgroup": e["subgroup"],
        "name_en": name_en,
        "name_zh": name_zh,
        "keywords_en": kwe,
        "keywords_zh": kwz,
        "emoji_version": e["emoji_version"],
        "status": e["status"],
        "png": ("images/png72/%s.png" % png) if png else "",
        "svg": ("images/svg/%s.svg" % svg) if svg else "",
    }
    records.append(rec)

groups_order, groups_count = [], {}
for r in records:
    if r["group"] not in groups_count:
        groups_order.append(r["group"])
        groups_count[r["group"]] = {"en": r["group"], "zh": r["group_zh"], "total": 0, "fq": 0}
    groups_count[r["group"]]["total"] += 1
    if r["status"] == "fully-qualified":
        groups_count[r["group"]]["fq"] += 1
groups = [groups_count[g] for g in groups_order]

fq = sum(1 for r in records if r["status"] == "fully-qualified")
meta = {
    "catalog_source": "Unicode emoji-test.txt 16.0",
    "names_source": "CLDR annotations (en/zh), Unicode-3.0",
    "images_source": "Twemoji v17.0.3 (graphics: CC-BY-4.0, code: MIT)",
    "generated": datetime.date.today().isoformat(),
    "total": len(records),
    "fully_qualified": fq,
    "with_chinese_name": sum(1 for r in records if r["name_zh"]),
    "matched_images": len(records) - missing_img,
}

with open(os.path.join(base, "emoji_catalog.json"), "w", encoding="utf-8") as f:
    json.dump({"meta": meta, "groups": groups, "emojis": records}, f, ensure_ascii=False, indent=1)

with open(os.path.join(base, "emoji_catalog.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["emoji", "codes", "group", "group_zh", "subgroup", "name_en", "name_zh",
                "keywords_en", "keywords_zh", "emoji_version", "status", "png", "svg"])
    for r in records:
        w.writerow([r["emoji"], r["codes"], r["group"], r["group_zh"], r["subgroup"],
                    r["name_en"], r["name_zh"], "; ".join(r["keywords_en"]),
                    "; ".join(r["keywords_zh"]), r["emoji_version"], r["status"], r["png"], r["svg"]])

# ---------------------------------------------------------------- HTML
gidx = {g: i for i, g in enumerate(groups_order)}
compact = [[r["emoji"], gidx[r["group"]], r["codes"], r["name_en"], r["name_zh"],
            ",".join(r["keywords_en"]), ",".join(r["keywords_zh"]),
            1 if r["status"] == "fully-qualified" else 0, r["png"]] for r in records]
garr = [[g["en"], g["zh"], g["fq"], g["total"]] for g in groups]

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Emoji 完整表情库</title>
<style>
*{box-sizing:border-box}
body{margin:0;font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;background:#f5f6fa;color:#222}
header{position:sticky;top:0;z-index:10;background:#fff;border-bottom:1px solid #e3e5ee;padding:12px 20px 8px;box-shadow:0 1px 6px rgba(20,25,40,.06)}
h1{margin:0 0 8px;font-size:19px}
h1 small{color:#888;font-weight:400;font-size:12px;margin-left:10px}
.toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:8px}
#q{flex:1 1 260px;padding:8px 12px;border:1px solid #cfd3e0;border-radius:8px;font-size:14px;outline:none}
#q:focus{border-color:#7c8cf8;box-shadow:0 0 0 3px rgba(124,140,248,.18)}
label.tog{font-size:13px;color:#555;display:flex;align-items:center;gap:4px;cursor:pointer;user-select:none}
#count{font-size:13px;color:#7a7f99;min-width:130px;text-align:right}
#tabs{display:flex;flex-wrap:wrap;gap:6px}
#tabs button{border:1px solid transparent;background:#eef0f7;color:#3b4256;padding:5px 12px;border-radius:16px;font-size:13px;cursor:pointer}
#tabs button.on{background:#4c5ff7;color:#fff}
main{padding:14px 20px 60px}
section{margin-bottom:22px}
section h2{font-size:15px;color:#333;border-left:4px solid #4c5ff7;padding-left:8px}
section h2 small{color:#999;font-weight:400;margin-left:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(62px,1fr));gap:6px}
.cell{aspect-ratio:1/1;display:flex;align-items:center;justify-content:center;background:#fff;border:1px solid #ececf3;border-radius:9px;cursor:pointer;position:relative;transition:transform .06s, box-shadow .06s}
.cell:hover{transform:translateY(-2px);box-shadow:0 4px 10px rgba(60,70,120,.14);border-color:#c9cdf5}
.cell img{width:34px;height:34px}
.cell .ch{font-size:29px;line-height:1;font-family:"Segoe UI Emoji","Apple Color Emoji","Noto Color Emoji",sans-serif}
.cell .nv{position:absolute;top:2px;right:4px;font-size:9px;color:#b6bacb}
#toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%) translateY(20px);background:#22243a;color:#fff;padding:10px 18px;border-radius:10px;font-size:14px;opacity:0;pointer-events:none;transition:all .18s;max-width:86vw;text-align:center;z-index:20}
#toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
#toast b{font-size:20px;margin-right:8px}
.empty{color:#98a;grid-column:1/-1;text-align:center;padding:40px 0;font-size:14px}
</style>
</head>
<body>
<header>
<h1>Emoji 完整表情库 <small id="stats"></small></h1>
<div class="toolbar">
<input id="q" placeholder="搜索：中英名称 / 关键词 / 编码，如：咧嘴、grin、1f600">
<label class="tog"><input type="checkbox" id="usefont">系统字体渲染</label>
<label class="tog"><input type="checkbox" id="showAll">显示非完全限定变体</label>
<span id="count"></span>
</div>
<nav id="tabs"></nav>
</header>
<main id="main"></main>
<div id="toast"></div>
<script>
const DATA=__DATA__;
const GROUPS=__GROUPS__;
let curGroup=-1;
const q=document.getElementById('q'),tabs=document.getElementById('tabs'),main=document.getElementById('main'),
count=document.getElementById('count'),toast=document.getElementById('toast'),
usefont=document.getElementById('usefont'),showAll=document.getElementById('showAll');
document.getElementById('stats').textContent='共 '+DATA.length+' 个（完全限定 '+DATA.filter(d=>d[7]).length+'）· Unicode 16.0 + Twemoji v17.0.3';
function render(){
  const kw=q.value.trim().toLowerCase();
  const frag=document.createDocumentFragment();
  GROUPS.forEach((g,gi)=>{
    if(curGroup>=0&&curGroup!==gi)return;
    const items=DATA.filter(d=>d[1]===gi&&(!kw||match(d,kw))&&(showAll.checked||d[7]));
    if(!items.length)return;
    const sec=document.createElement('section');
    const h=document.createElement('h2');
    h.innerHTML=g[1]+' '+g[0]+'<small>'+items.length+'</small>';
    sec.appendChild(h);
    const grid=document.createElement('div');grid.className='grid';
    items.forEach(d=>{
      const c=document.createElement('div');c.className='cell';c.title=d[4]||d[3];
      c.innerHTML=usefont.checked?'<span class="ch">'+d[0]+'</span>'
        :'<img loading="lazy" src="'+d[8]+'" alt="'+(d[4]||d[3])+'" onerror="this.outerHTML=\'<span class=ch>'+d[0]+'</span>\'">';
      if(!d[7]){const n=document.createElement('span');n.className='nv';n.textContent='V';c.appendChild(n);}
      c.onclick=()=>pick(d);
      grid.appendChild(c);
    });
    if(!grid.children.length){const e=document.createElement('div');e.className='empty';e.textContent='无匹配结果';grid.appendChild(e);}
    sec.appendChild(grid);frag.appendChild(sec);
  });
  main.innerHTML='';main.appendChild(frag);
  count.textContent='显示 '+main.querySelectorAll('.cell').length+' 个';
}
function match(d,kw){
  return d[0].indexOf(kw)>=0||d[3].toLowerCase().indexOf(kw)>=0||(d[4]||'').indexOf(kw)>=0
    ||d[2].indexOf(kw)>=0||d[5].toLowerCase().indexOf(kw)>=0||(d[6]||'').indexOf(kw)>=0;
}
let tt;
function pick(d){
  const txt=d[0];
  (navigator.clipboard?navigator.clipboard.writeText(txt):Promise.reject()).then(ok,()=>fallback());
  function fallback(){const t=document.createElement('textarea');t.value=txt;document.body.appendChild(t);t.select();try{document.execCommand('copy')}catch(e){}t.remove();ok();}
  function ok(){
    toast.innerHTML='<b>'+txt+'</b> 已复制 · '+(d[4]||'—')+' · '+(d[3]||'—')+' · '+d[2];
    toast.classList.add('show');clearTimeout(tt);tt=setTimeout(()=>toast.classList.remove('show'),1600);
  }
}
function buildTabs(){
  let html='<button data-g="-1" class="'+(curGroup<0?'on':'')+'">全部</button>';
  GROUPS.forEach((g,i)=>{html+='<button data-g="'+i+'" class="'+(curGroup===i?'on':'')+'">'+g[1]+' '+g[3]+'</button>';});
  tabs.innerHTML=html;
  tabs.querySelectorAll('button').forEach(b=>b.onclick=()=>{curGroup=+b.dataset.g;buildTabs();render();});
}
let deb;q.addEventListener('input',()=>{clearTimeout(deb);deb=setTimeout(render,120);});
usefont.onchange=render;showAll.onchange=render;
buildTabs();render();
</script>
</body>
</html>
"""

html = HTML.replace("__DATA__", json.dumps(compact, ensure_ascii=False, separators=(",", ":")))
html = html.replace("__GROUPS__", json.dumps(garr, ensure_ascii=False, separators=(",", ":")))
with open(os.path.join(base, "index.html"), "w", encoding="utf-8") as f:
    f.write(html)

# ---------------------------------------------------------------- report
print("total entries        :", len(records))
print("fully-qualified      :", fq)
print("with zh name         :", meta["with_chinese_name"])
print("matched png images   :", len(records) - missing_img, " missing:", missing_img)
print("groups               :", len(groups))
for g in groups:
    print("   %-18s %-8s fq=%d total=%d" % (g["en"], g["zh"], g["fq"], g["total"]))
