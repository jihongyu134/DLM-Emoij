# 情感对话 Emoji 训练数据集 + 完整 Emoji 表情库

中英双语 LLM 微调（SFT）数据集：教会模型**在情感陪伴场景自然使用 emoji、在普通问答场景保持中性不用 emoji**。另附完整 Unicode emoji 表情库（名称 + 图片 + 可搜索浏览器）。

## 目录结构

```
D:\情感对话emoij\
├── data\                        # 最终训练集（由脚本生成）
│   ├── zh\train.jsonl / val.jsonl
│   ├── en\train.jsonl / val.jsonl
│   └── stats.json               # 校验与配额报告
├── raw\                         # 8个写作批次原始数据（校验前）
├── scripts\
│   ├── validate_and_split.py    # 校验+去重+配额抽样+train/val切分
│   ├── train_tiny_gpt.py        # 自研微型GPT从零训练（~87M）
│   ├── chat_tiny.py             # 本地对话测试
│   ├── dashboard.py             # 实时面板 (http://127.0.0.1:8799)
│   └── build_emoji_catalog.py   # 构建 emoji 表情库目录
├── emoji_library\               # 完整 emoji 表情库 ★
│   ├── index.html               # 可搜索/点击复制的离线浏览器
│   ├── emoji_catalog.json       # 5042条结构化目录
│   ├── emoji_catalog.csv        # Excel 友好 CSV（utf-8-sig）
│   ├── images\png72\ (4009张)   # Twemoji v17.0.3 72x72 PNG
│   ├── images\svg\   (4009张)   # Twemoji v17.0.3 SVG
│   ├── licenses\                # CC-BY-4.0 图形许可
│   └── src\                     # Unicode/CLDR 源数据
├── dataset_info.json            # LLaMA-Factory 数据集注册
└── README.md
```

## 数据集规格（最终版）

- **规模**：500 条（zh 250 / en 250），每条 2~4 轮多轮对话
- **切分**：train 450 条（每语言 226）/ val 50 条（每语言 24）
- **情绪配额**（每种语言，全部精确命中）：joy 45 / sadness 40 / anger 35 / fear 30 / surprise 30 / disgust 25 / neutral 45
- **格式**（messages 版）：
```json
{"system": "你是一个温暖贴心的聊天伙伴……", "messages": [{"role": "user", "content": "我今天升职了！"}, {"role": "assistant", "content": "哇！太厉害了🎉 恭喜你！"}], "emotion": "joy", "lang": "zh", "topic": "职场"}
```
- **核心规则**（脚本强制校验）：
  - 情感类：assistant 每条回复 1~3 个 emoji，贴合情绪（悲伤不用庆祝类等）
  - neutral：任何消息零 emoji，语气专业平实
- **切分**：每语言每情绪约 10% 进 val，其余 train

## 自研模型：从零训练（不使用预训练模型微调）

`scripts\train_tiny_gpt.py` 是一套纯自研的微型 GPT 训练方案（仅依赖 PyTorch，无任何预训练权重）：

```bash
python -I scripts/train_tiny_gpt.py --raw-dir data --out-dir models/tiny_emoji_gpt_final --steps 3500
```

- **最终模型**：`models/tiny_emoji_gpt_final/ckpt.pt`（87.2M 参数，best train loss 0.134，69 分钟训完）
- **效果验证**：情感话题自动带 1~3 个贴合情绪的 emoji；普通问答零 emoji 纯文本

- **架构**：86.9M 参数（12层 / 768维 / 12头），字符级 tokenizer（自动从语料构建，含 emoji）
- **SFT 目标**：仅对 assistant 回复计算 loss，user/system 仅作条件输入
- **优化**：AdamW + 余弦退火 + bf16 混合精度，RTX 3050 约 40 分钟训完
- **对话测试**：

```bash
python -I scripts/chat_tiny.py                    # 自动判断情感/中性人格
python -I scripts/chat_tiny.py --mood emotional   # 强制带emoji陪伴模式
python -I scripts/chat_tiny.py --mood neutral     # 强制普通助手模式
```

## Emoji 表情库

打开 `emoji_library\index.html` 即可离线浏览全部 emoji：
- 中/英文名称与关键词搜索（如"咧嘴"、"grin"、"1f600"）
- 按 10 大分类过滤；点击复制；可切换系统字体/Twemoji 图片渲染
- 完全限定变体默认展示（可开关显示非标准变体）

数据来源与许可：
| 内容 | 来源 | 许可 |
|---|---|---|
| emoji 清单 (5042条) | Unicode emoji-test.txt 16.0 | Unicode-3.0 |
| 中英名称/关键词 | CLDR 49.0 annotations + annotationsDerived | Unicode-3.0 |
| 图片 (PNG/SVG) | Twemoji v17.0.3 (jdecked) | CC-BY-4.0 |

## 重建命令

```bash
python -I scripts/build_emoji_catalog.py emoji_library   # 重建 emoji 目录
python -I scripts/validate_and_split.py .                # 校验+切分（需 raw\*.jsonl 齐）
python -I scripts/train_tiny_gpt.py                      # 自研模型从零训练
```
