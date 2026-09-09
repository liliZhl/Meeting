# -*- coding: utf-8 -*-
"""把 1024 PNG 转多尺寸 .ico（含 16/32/48/64/128/256 满足 Windows 任务栏/资源管理器）。"""
import sys
from PIL import Image

src = sys.argv[1]
dst = sys.argv[2]
img = Image.open(src).convert("RGBA")
# 防止纯白背景被 OS 误裁：用圆形 alpha mask 把四个角变透明
mask = Image.new("L", img.size, 0)
from PIL import ImageDraw
draw = ImageDraw.Draw(mask)
w, h = img.size
radius = int(min(w, h) * 0.22)
draw.rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
img.putalpha(mask)

sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
img.save(dst, format="ICO", sizes=sizes)
print(f"WROTE {dst} (sizes={sizes})")
