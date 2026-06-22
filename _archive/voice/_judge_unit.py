# -*- coding: utf-8 -*-
"""parse_judge 离线单测(不连机器人,不连 API)。"""
import sys

sys.path.insert(0, r"D:\workspace\reachy-mini\voice")
# 只导函数,避免 d01 顶层连接副作用:按文本抽取
import json
import re

src = open(r"D:\workspace\reachy-mini\voice\d01_realtime_chat.py", encoding="utf-8").read()
m = re.search(r"def parse_judge.*?(?=\n\n|\ndef |\nclass )", src, re.S)
ns = {"json": json}
exec(m.group(0), ns)
parse_judge = ns["parse_judge"]

cases = [
    ('裸JSON', '{"pointing": true, "target_visible": false, "direction": "右", "desc": ""}'),
    ('代码块', '```json\n{"pointing": false, "target_visible": false, "direction": "无", "desc": "画面里有一个人"}\n```'),
    ('带前后缀', '好的,这是结果:{"pointing": true, "target_visible": true, "direction": "左", "desc": "他指的是水杯"} 以上。'),
    ('垃圾', '画面中有一位男士坐在椅子上。'),
    ('坏JSON', '{"pointing": true, "direction": "右"'),
]
for name, raw in cases:
    print(f"{name:6s} → {parse_judge(raw)}")
