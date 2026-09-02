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

# scipy / numpy 子模块（不再用 collect_submodules 全量收集——上面 collect_all 已含）
# 去重
hiddenimports = list(dict.fromkeys(hiddenimports))
binaries = list({b[0]: b for b in binaries}.values())
datas = list({d[0]: d for d in datas}.values())

a = Analysis(
    ["src/main.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
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
    icon=None,
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
