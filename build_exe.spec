# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置（onedir 模式）
智能会议纪要工具

关键点：
  - funasr / modelscope / torch 等为延迟动态导入，需显式 collect_all
  - 模型文件(mod/)与 ffmpeg(bin/) 不打包，运行时从 EXE 同级目录加载
  - remote_code 加载的 model.py/ctc.py/tools 在模型目录内，运行时 sys.path 处理
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_dynamic_libs

import glob
import os

datas = []
binaries = []
hiddenimports = []

# 收集关键库（datas + binaries + hiddenimports 三合一）
_collect_pkgs = [
    "funasr",
    "modelscope",
    "librosa",
    "numba",
    "transformers",
    "soundfile",
    "tokenizers",
    "tiktoken",
    "kaldiio",
    "omegaconf",
    "hydra",
    "jieba",
    "sentencepiece",
    "soxr",
    "audioread",
    "pydub",
    "pyaudio",
    "scipy",
    "sklearn",
    "umap",
    "pynndescent",
    "llvmlite",
    "numpy",
]

for pkg in _collect_pkgs:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
        print(f"[spec] collect_all('{pkg}') OK: {len(d)} datas, {len(b)} bins, {len(h)} hidden")
    except Exception as e:
        print(f"[spec] collect_all('{pkg}') FAILED: {e}")

# 手动补充常见隐藏导入
hiddenimports += [
    "asr_engine",
    "model_manager",
    "player",
    "store",
    "theme",
    "funasr.register",
    "funasr.auto.auto_model",
    "funasr.models.fun_asr_nano.checkpoint_utils",
    "funasr.models.fun_asr_nano.device_utils",
    "funasr.utils.dynamic_import",
    "funasr.download.download_model_from_hub",
    "funasr.utils.load_utils",
    "funasr.utils.install_model_requirements",
    "modelscope",
    "torch",
    "torchaudio",
    "torchaudio.compliance.kaldi",
    "soundfile",
    "numba.core",
    "llvmlite.binding",
]

# ── 加速优化：过滤 collect_all/collect_submodules 引入的测试类模块 ──
# 实测 6634 个 hiddenimports 中 1228 个(19%)是 tests/test_* 子模块，
# 运行时根本不会被 import，却让分析阶段逐行扫描拖慢打包。
# numba(55%)/sklearn(45%)/scipy(35%)/numpy(30%) 是重灾区。
def _is_test_module(mod: str) -> bool:
    segs = mod.split(".")
    return any(s in ("tests", "test") or s.startswith("test_") or s.endswith("_test") for s in segs)

_before = len(hiddenimports)
hiddenimports = [m for m in hiddenimports if not _is_test_module(m)]
print(f"[spec] 过滤测试模块: {_before} -> {len(hiddenimports)} (省 {_before - len(hiddenimports)})")

# torch 动态库（CUDA 相关）
try:
    torch_bins = collect_dynamic_libs("torch")
    binaries += torch_bins
    print(f"[spec] collect_dynamic_libs('torch') OK: {len(torch_bins)} bins")
except Exception as e:
    print(f"[spec] collect_dynamic_libs('torch') FAILED: {e}")

# ── 阶段 F：QtMultimedia 播放支持 ─────────────────────────────────────
# 1) Python 模块 + Qt6Multimedia DLL
try:
    d, b, h = collect_all("PyQt6.QtMultimedia")
    datas += d
    binaries += b
    hiddenimports += h
    print(f"[spec] collect_all('PyQt6.QtMultimedia') OK: {len(d)} datas, {len(b)} bins, {len(h)} hidden")
except Exception as e:
    print(f"[spec] collect_all('PyQt6.QtMultimedia') FAILED: {e}")

# 2) Qt6 多媒体后端插件（ffmpeg / windows）。PyInstaller 官方 PyQt6 hook 不收集，
#    上一版 EXE 因缺此目录导致播放完全不可用（dist 内 plugins 下无 multimedia）。
try:
    import PyQt6 as _PyQt6
    _qt6_dir = os.path.join(os.path.dirname(_PyQt6.__file__), "Qt6")
    _mm_plugin_dir = os.path.join(_qt6_dir, "plugins", "multimedia")
    _n = 0
    for src in glob.glob(os.path.join(_mm_plugin_dir, "*.dll")):
        binaries.append((src, "PyQt6/Qt6/plugins/multimedia"))
        _n += 1
    print(f"[spec] multimedia 插件收集: {_n} 个 -> {glob.glob(os.path.join(_mm_plugin_dir, '*.dll'))}")
except Exception as e:
    print(f"[spec] multimedia 插件收集 FAILED: {e}")

# 3) ffmpeg 解码 DLL（Qt 自带 ffmpeg 后端运行时动态加载，非链接依赖，需手动带上）
try:
    _n = 0
    for pattern in ("av*.dll", "sw*.dll"):
        for src in glob.glob(os.path.join(_qt6_dir, "bin", pattern)):
            binaries.append((src, "PyQt6/Qt6/bin"))
            _n += 1
    print(f"[spec] ffmpeg 解码 DLL 收集: {_n} 个")
except Exception as e:
    print(f"[spec] ffmpeg DLL 收集 FAILED: {e}")

# scipy / numpy 子模块（不再用 collect_submodules 全量收集——上面 collect_all 已含）
# 去重
hiddenimports = list(dict.fromkeys(hiddenimports))
binaries = list({b[0]: b for b in binaries}.values())
datas = list({d[0]: d for d in datas}.values())

a = Analysis(
    ["src/main.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas + [("src/app.ico", ".")],   # 2026-09-09：运行期窗口/任务栏图标随包
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "notebook",
        "pytest",
        "PySide6",
        "PyQt5",
        "torchvision",
        "torchtext",
        "tensorflow",
        "jupyter",
        "setuptools",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MeetingAssistant",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # windowed（无控制台窗口）
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="src/app.ico",       # 2026-09-09：EXE 文件图标（资源管理器/任务栏）
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MeetingAssistant",
)
