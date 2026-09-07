# -*- coding: utf-8 -*-
"""
智能会议纪要工具 - 主程序（Windows 桌面应用）
============================================
功能：
  1. 麦克风录音 / 音频文件导入
  2. 语音转文字（Fun-ASR-Nano + 说话人分离，进程内推理）
  3. AI 智能总结（DeepSeek API）
  4. 会议纪要导出（txt）
  5. 配置管理（API Key / 主题 / 模型目录）

技术栈：PyQt6 + pyaudio + pydub + funasr + openai
架构：进程内 AutoModel 推理（非 server 子进程），便于打包分发。
"""

import os
import sys
import json
import wave
import time
import shutil
import logging
import threading
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. 路径与运行环境检测（兼容 PyInstaller 打包后的 frozen 环境）
# ---------------------------------------------------------------------------

def is_frozen() -> bool:
    """判断是否为 PyInstaller 打包后的可执行程序。"""
    return getattr(sys, "frozen", False)


def get_app_dir() -> Path:
    """
    获取应用根目录。
    - 打包后（onedir）：EXE 所在目录
    - 开发时：源码目录
    """
    if is_frozen():
        return Path(sys.executable).parent
    return Path(__file__).parent


APP_DIR = get_app_dir()
CONFIG_PATH = APP_DIR / "config.json"
LOG_DIR = APP_DIR / "logs"
MODEL_DIR = APP_DIR / "mod"                     # 模型目录（不打包，分发时一起拷）
BIN_DIR = APP_DIR / "bin"                       # ffmpeg.exe 所在目录
FFMPEG_PATH = BIN_DIR / "ffmpeg.exe"
RECORDS_DIR = APP_DIR / "records"               # 历史记录（阶段 B）

# 确保关键目录存在
LOG_DIR.mkdir(parents=True, exist_ok=True)
BIN_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. 日志系统（调试关键：记录到文件 + 控制台，含点击等详细事件）
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """初始化日志，返回根 logger。"""
    logger = logging.getLogger("MeetingAssistant")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file = LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 打包成 windowed 后无控制台，stdout 可能为 None，做保护
    try:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    except Exception:
        pass

    logger.info("=" * 70)
    logger.info("智能会议纪要工具 启动")
    logger.info(f"应用目录: {APP_DIR}")
    logger.info(f"是否打包运行: {is_frozen()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"日志文件: {log_file}")
    return logger


logger = setup_logging()


# ---------------------------------------------------------------------------
# 1.5 主题（阶段 E：样式集中在 theme.py，此处仅取配色常量）
# ---------------------------------------------------------------------------
try:
    from theme import build_qss, get_colors
    THEME_OK = True
except Exception as _e:                                    # pragma: no cover
    THEME_OK = False
    logger.warning(f"theme.py 加载失败，将使用内置兜底配色: {_e}")

    def build_qss(mode: str = "dark") -> str:               # type: ignore
        return ""

    def get_colors(mode: str = "dark") -> dict:             # type: ignore
        return {
            "ts": "#7aa2f7", "spk": "#9ece6a", "body": "#c0caf5",
            "highlight": "255,200,60,70", "success": "#9ece6a", "accent": "#7aa2f7",
        }


# ---------------------------------------------------------------------------
# 2. 依赖导入（优雅降级：缺失时记录警告，界面仍可启动）
# ---------------------------------------------------------------------------

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QTextEdit, QLabel, QFileDialog, QMessageBox, QDialog,
        QLineEdit, QComboBox, QDialogButtonBox, QProgressDialog, QFrame, QStatusBar,
        QProgressBar, QFormLayout, QPlainTextEdit, QTextBrowser, QSplitter,
        QListWidget, QListWidgetItem, QAbstractItemView, QInputDialog, QMenu,
        QSlider, QStyle, QStyleOptionSlider,
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject, QEvent, QUrl
    from PyQt6.QtGui import (
        QFont, QTextCursor, QCloseEvent, QTextCharFormat, QColor, QCursor,
    )
    PYQT_OK = True

    # 播放功能依赖 QtMultimedia（阶段 C）
    try:
        # 后端策略（阶段 F 调整）：
        #   Qt 6.11 自带 ffmpegmediaplugin，对 16k PCM wav 解码最稳、无系统解码器依赖，
        #   因此不再强制 windows(WMF) 后端，交给 Qt 自己选择。
        #   需要切回 WMF 时设置环境变量 QT_MEDIA_BACKEND=windows 即可覆盖。
        from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput  # noqa: F401
        QT_MULTIMEDIA_OK = True
    except Exception:
        QT_MULTIMEDIA_OK = False
        logger.warning("QtMultimedia 不可用，播放功能将隐藏")
except ImportError as e:
    PYQT_OK = False
    logger.critical(f"PyQt6 导入失败：{e}")

try:
    import pyaudio
    PYAUDIO_OK = True
except ImportError:
    PYAUDIO_OK = False
    pyaudio = None
    logger.warning("pyaudio 未安装，麦克风录音功能不可用")

try:
    from pydub import AudioSegment
    PYDUB_OK = True
except ImportError:
    PYDUB_OK = False
    AudioSegment = None
    logger.warning("pydub 未安装，音频格式转换功能不可用")

try:
    from openai import OpenAI
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False
    OpenAI = None
    logger.warning("openai 库未安装，总结功能不可用")

def _locate_ffmpeg():
    """多级定位 ffmpeg.exe（2026-09-07 修复：开发目录无 bin/ 时导入 mp3 必失败）。

    候选顺序：打包/应用 bin → 可执行文件旁 bin → 系统 PATH。
    返回可执行文件绝对路径或 None（None 时 mp3/m4a/flac 转换受限，pydub 仍可能自行找到系统 ffmpeg）。
    """
    cands = [
        FFMPEG_PATH,                                              # APP_DIR/bin/ffmpeg.exe
        Path(sys.executable).parent / "bin" / "ffmpeg.exe",       # frozen 时 EXE 旁 bin
    ]
    for p in cands:
        try:
            if p.exists():
                return str(p)
        except Exception:
            pass
    w = shutil.which("ffmpeg")
    return w


FFMPEG_EXE = _locate_ffmpeg()
FFMPEG_AVAILABLE = FFMPEG_EXE is not None
if not FFMPEG_AVAILABLE:
    logger.warning("未定位到 ffmpeg.exe：mp3/m4a/flac 导入转换将受限（可把 ffmpeg.exe 放到 bin/ 或加入 PATH）")


# ---------------------------------------------------------------------------
# 3. 配置管理
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "deepseek_api_key": "",
    "deepseek_base_url": "https://api.deepseek.com",
    "deepseek_model": "deepseek-chat",
    "theme": "light",
    "summary_template": "meeting",
    "model_dir": "mod",           # 模型目录（相对 EXE 或绝对路径）
    "asr_model": "sensevoice",  # 识别模型: sensevoice(默认) / fun-asr-nano / paraformer
    "vad_level": "medium",         # VAD 切句灵敏度: fine / medium / coarse
}


class ConfigManager:
    """配置文件读写。config.json 位于 EXE 同级目录。"""

    def __init__(self, path: Path = CONFIG_PATH):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"已加载配置文件: {self.path}")
                return {**DEFAULT_CONFIG, **data}
            except Exception as e:
                logger.error(f"配置文件解析失败，使用默认值：{e}")
        logger.info("配置文件不存在，将使用默认值")
        return dict(DEFAULT_CONFIG)

    def save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=4)
            logger.info(f"配置已保存: {self.path}")
        except Exception as e:
            logger.error(f"配置保存失败: {e}")

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


# ---------------------------------------------------------------------------
# 4. 全局鼠标点击日志（调试关键）
# ---------------------------------------------------------------------------

class ClickLogger(QObject):
    """全局事件过滤器，记录鼠标点击详情。"""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            pos = event.position()
            name = obj.objectName() or "(未命名)"
            logger.debug(
                f"[点击按下] {obj.__class__.__name__} '{name}' "
                f"({pos.x():.0f},{pos.y():.0f}) 按钮={event.button()}"
            )
        elif event.type() == QEvent.Type.MouseButtonRelease:
            pos = event.position()
            name = obj.objectName() or "(未命名)"
            logger.debug(
                f"[点击释放] {obj.__class__.__name__} '{name}' "
                f"({pos.x():.0f},{pos.y():.0f}) 按钮={event.button()}"
            )
        return super().eventFilter(obj, event)


# ---------------------------------------------------------------------------
# 5. 工具函数
# ---------------------------------------------------------------------------

def format_timestamp(seconds) -> str:
    """秒 -> HH:MM:SS。"""
    if seconds is None:
        return "00:00:00"
    try:
        s = int(float(seconds))
    except (TypeError, ValueError):
        return "00:00:00"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def convert_to_wav_16k(input_path: str) -> str:
    """任意音频 -> 16kHz 单声道 WAV，返回临时文件路径。"""
    src = Path(input_path)
    logger.info(f"开始转换音频: {src.name}")

    # 已满足条件则直接返回
    if src.suffix.lower() == ".wav":
        try:
            with wave.open(str(src), "rb") as wf:
                if wf.getframerate() == 16000 and wf.getnchannels() == 1:
                    logger.info("音频已是 16kHz 单声道 WAV")
                    return str(src)
        except Exception:
            pass

    if PYDUB_OK:
        try:
            if FFMPEG_EXE:
                AudioSegment.converter = FFMPEG_EXE
                AudioSegment.ffmpeg = FFMPEG_EXE
                AudioSegment.ffprobe = FFMPEG_EXE
            audio = AudioSegment.from_file(str(src))
            audio = audio.set_frame_rate(16000).set_channels(1)
            tmp = Path(tempfile.gettempdir()) / f"meeting_conv_{int(time.time()*1000)}.wav"
            audio.export(str(tmp), format="wav")
            logger.info(f"音频转换完成(pydub): {tmp}")
            return str(tmp)
        except Exception as e:
            logger.error(f"pydub 转换失败: {e}")

    if FFMPEG_AVAILABLE:
        try:
            tmp = Path(tempfile.gettempdir()) / f"meeting_conv_{int(time.time()*1000)}.wav"
            cmd = [FFMPEG_EXE, "-y", "-i", str(src),
                   "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(tmp)]
            logger.info(f"执行 ffmpeg: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"音频转换完成(ffmpeg): {tmp}")
            return str(tmp)
        except Exception as e:
            logger.error(f"ffmpeg 转换失败: {e}")

    raise RuntimeError("音频转换失败：需要 pydub 或 bin/ffmpeg.exe")


# ---------------------------------------------------------------------------
# 6. 录音线程
# ---------------------------------------------------------------------------

class AudioRecorder(QThread):
    """麦克风录音（16kHz 单声道 16bit PCM WAV）。"""
    started = pyqtSignal()
    stopped = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, sample_rate=16000, channels=1):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self._stop = threading.Event()
        self.frames = []

    def run(self):
        pa = None
        stream = None
        try:
            pa = pyaudio.PyAudio()
            info = pa.get_default_input_device_info()
            logger.info(f"默认麦克风: {info.get('name')}")
            stream = pa.open(format=self.format, channels=self.channels,
                             rate=self.sample_rate, input=True,
                             frames_per_buffer=self.chunk)
            self.started.emit()
            logger.info("录音开始")
            while not self._stop.is_set():
                data = stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
            logger.info("录音循环结束")
            path = self._save_wav()
            self.stopped.emit(path)
        except Exception as e:
            logger.exception("录音失败")
            self.error.emit(str(e))
        finally:
            if stream is not None:
                try:
                    stream.stop_stream(); stream.close()
                except Exception:
                    pass
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    pass

    def stop(self):
        self._stop.set()

    def _save_wav(self) -> str:
        rec_dir = APP_DIR / "recordings"
        rec_dir.mkdir(parents=True, exist_ok=True)
        filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        path = rec_dir / filename
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(pyaudio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(self.frames))
        logger.info(f"录音已保存: {path}")
        return str(path)


# ---------------------------------------------------------------------------
# 7. 转写线程（进程内 ASR 引擎 + 说话人分离）
# ---------------------------------------------------------------------------

class TranscriptionWorker(QThread):
    """语音转写（含说话人分离）。

    模型由 ModelManager 单例常驻管理（启动时后台预加载），
    转写时直接复用，不再每次新建引擎重新加载。
    """
    finished = pyqtSignal(str)          # 纯文本转写结果（兼容当前 UI）
    finished_obj = pyqtSignal(object)   # 结构化结果 TranscriptionResult
    error = pyqtSignal(str)
    progress = pyqtSignal(str)    # 阶段状态文本

    def __init__(self, audio_path: str, model_dir: str, audio_name: str = "",
                 asr_model: str = None, vad_level: str = None,
                 num_speakers: int = None):
        super().__init__()
        self.audio_path = audio_path
        self.model_dir = model_dir
        self.audio_name = audio_name or Path(audio_path).name
        self.asr_model = asr_model
        self.vad_level = vad_level
        self.num_speakers = num_speakers

    def run(self):
        try:
            from model_manager import get_manager
            # 单例 manager：若后台尚未加载完成则同步等待
            manager = get_manager(model_root=self.model_dir, asr_model=self.asr_model,
                                  vad_level=self.vad_level)
            if not manager.is_ready():
                self.progress.emit("正在加载语音模型（首次约需 1 分钟，请耐心等待）…")
            else:
                self.progress.emit("模型已就绪，开始转写…")
            if self.num_speakers and self.num_speakers > 1:
                self.progress.emit(f"预设 {self.num_speakers} 个说话人，正在转写…")
            else:
                self.progress.emit("正在转写（说话人数自动估计），音频较长时可能需要几分钟…")
            result = manager.transcribe(
                self.audio_path, progress_callback=self._on_progress,
                num_speakers=self.num_speakers,
            )
            logger.info("转写完成")
            # 结构化结果与纯文本同时发出，UI 按需取用
            self.finished_obj.emit(result)
            text = result.to_text() if hasattr(result, "to_text") else str(result)
            self.finished.emit(text or "(无转写结果)")
        except Exception as e:
            logger.exception("转写失败")
            self.error.emit(str(e))

    def _on_progress(self, step: str, detail: str = ""):
        self.progress.emit(f"{step}：{detail}" if detail else step)


# ---------------------------------------------------------------------------
# 8. 总结线程（DeepSeek API）
# ---------------------------------------------------------------------------

# 总结模板（2026-09-07 新增多模板：会议/通话/待办/要点；可在工具栏下拉切换）
SUMMARY_TEMPLATES = {
    "meeting": (
        "会议纪要",
        """你是一名专业的会议纪要助理。请根据以下转录文本（含说话人和时间戳）生成结构化的会议纪要。

要求：
1. 用中文输出
2. 严格按照下面的 Markdown 结构
3. 转录中没有的信息标注"未提及"，不要编造
4. 结合说话人标签归纳每个人的发言要点

## 会议概要
- 主题：[一句话概括]
- 参会人员：[根据说话人标签列出，如"说话人1、说话人2"；有真实姓名则用姓名]

## 核心讨论与决策
- [议题]：[讨论要点与最终决策]

## 行动项
| 负责人 | 待办事项 | 截止时间 |
|--------|----------|----------|
| 未提及 | 未提及 | 未提及 |

## 遗留问题
- [列出未解决的问题，没有则写"无"]

转录文本：
{transcript}""",
    ),
    "call": (
        "通话摘要",
        """你是一名专业的通话摘要助理。请根据以下转录文本（含说话人和时间戳）整理一段简洁的通话纪要。

要求：
1. 用中文输出
2. 按下面的 Markdown 结构组织
3. 通话中没有的信息标注"未提及"，不要编造
4. 重点提炼"双方达成的共识"与"下一步约定"，便于事后跟进

## 通话概要
- 双方：[说话人标签；若有真实姓名则使用]
- 主要事项：[一句话概括这次通话讨论的核心内容]

## 关键要点
- [逐条列出通话中讨论/确认的重要事项]

## 共识与结论
- [双方明确达成一致的内容]

## 后续行动 / 约定
| 负责人 | 约定事项 | 时间/期限 |
|--------|----------|-----------|
| 未提及 | 未提及 | 未提及 |

## 待澄清 / 遗留
- [未解决或需后续确认的问题，没有则写"无"]

转录文本：
{transcript}""",
    ),
    "todos": (
        "任务/待办提取",
        """你是一名任务梳理助理。请根据以下转录文本（含说话人和时间戳）把所有"待办/行动项"提炼成结构化清单。

要求：
1. 用中文输出
2. 只输出明确的待办（隐含或猜测的内容写"未提及"）
3. 转录中无任何待办时，输出"未发现明确的待办事项"
4. 严格使用下面的 Markdown 表格

## 待办清单
| # | 待办事项 | 负责人 | 截止时间 | 依据/上下文 |
|---|----------|--------|----------|------------|
| 1 | ... | ... | ... | 引自转录中相关发言 |

## 跟进建议
- [可选：对紧迫待办/无负责人的待办给出提醒]

转录文本：
{transcript}""",
    ),
    "knowledge": (
        "知识/要点提炼",
        """你是一名内容提炼助理。请根据以下转录文本（含说话人和时间戳）将其浓缩为结构化的知识要点。

要求：
1. 用中文输出
2. 按下面的 Markdown 结构组织
3. 提炼要点，不要逐字复述
4. 转录中没有的内容标注"未提及"

## 内容概要
- [一句话总览转录主题]

## 核心要点
- [分条列出 3-7 条核心信息]

## 关键概念 / 术语
- [若涉及专业名词，列出来并简释]

## 典型引述
- [引用 1-3 条原话佐证要点；用 Markdown blockquote]

## 可深入方向
- [对想继续学习的读者给出建议]

转录文本：
{transcript}""",
    ),
}

DEFAULT_SUMMARY_TEMPLATE = "meeting"


class SummaryWorker(QThread):
    """AI 智能总结。"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, transcript, api_key, base_url, model, prompt_text):
        super().__init__()
        self.transcript = transcript
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.prompt_text = prompt_text

    def run(self):
        try:
            if not OPENAI_OK:
                raise RuntimeError("openai 库未安装")
            if not self.api_key:
                raise RuntimeError("DeepSeek API Key 为空")
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            prompt = self.prompt_text.format(transcript=self.transcript)
            logger.info(f"调用 DeepSeek 总结 (model={self.model})")
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一名专业的内容助理。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            content = resp.choices[0].message.content or ""
            logger.info("总结完成")
            self.finished.emit(content)
        except Exception as e:
            logger.exception("总结失败")
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# 8.5 无边框弹窗基类（2026-09-07：全部弹窗统一为无边框圆角卡片风格）
# ---------------------------------------------------------------------------

class NfDialog(QDialog):
    """无边框圆角弹窗基类。

    结构：Frameless + 透明背景 → 外层 #dlgCard（圆角面板色卡片）
    内：标题条（拖动区 + 标题 + ✕）→ self.body（子类填充内容）。
    拖动/双击关闭按钮由基类处理。
    """

    def __init__(self, title: str, parent=None, width: int = 460):
        super().__init__(parent)
        self.setObjectName("nfDlg")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._drag_off = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.card = QWidget()
        self.card.setObjectName("dlgCard")
        outer.addWidget(self.card)
        cvb = QVBoxLayout(self.card)
        cvb.setContentsMargins(14, 8, 14, 14)
        cvb.setSpacing(8)

        # 标题条（独立 widget 固定高度，避免被内容挤压）
        title_bar = QWidget()
        title_bar.setObjectName("dlgTitleBar")
        title_bar.setFixedHeight(34)
        tb = QHBoxLayout(title_bar)
        tb.setContentsMargins(2, 0, 4, 0)
        tb.setSpacing(4)
        self.lbl_dlg_title = QLabel(title)
        self.lbl_dlg_title.setObjectName("dlgTitle")
        tb.addWidget(self.lbl_dlg_title)
        tb.addStretch(1)
        self.btn_dlg_close = QPushButton("✕")
        self.btn_dlg_close.setObjectName("dlgClose")
        self.btn_dlg_close.setFixedSize(26, 22)
        self.btn_dlg_close.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_dlg_close.setToolTip("关闭")
        self.btn_dlg_close.clicked.connect(self.reject)
        tb.addWidget(self.btn_dlg_close)
        cvb.addWidget(title_bar)

        self.body = QWidget()
        self.body.setObjectName("dlgBody")
        self.body.setStyleSheet("QWidget#dlgBody { background: transparent; }")
        self.body_lay = QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(0, 2, 0, 0)
        self.body_lay.setSpacing(8)
        cvb.addWidget(self.body, 1)

        # 标题条拖动
        self.card.mousePressEvent = self._nf_press
        self.card.mouseMoveEvent = self._nf_move
        self.card.mouseReleaseEvent = self._nf_release
        self.setMinimumWidth(width)

    def _nf_press(self, e):
        if e.button() == Qt.MouseButton.LeftButton and not self.isMaximized():
            self._drag_off = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
        else:
            self._drag_off = None

    def _nf_move(self, e):
        off = self._drag_off
        if off is not None and (e.buttons() & Qt.MouseButton.LeftButton):
            self.move(e.globalPosition().toPoint() - off)

    def _nf_release(self, e):
        self._drag_off = None

    def exec(self):
        return super().exec()

    def _center_on(self, parent):
        """有父窗口时居中显示。"""
        if parent is not None:
            try:
                pg = parent.frameGeometry()
                self.adjustSize()
                self.move(pg.center().x() - self.width() // 2,
                          pg.center().y() - self.height() // 2)
            except Exception:
                pass


class NfMessage(NfDialog):
    """无边框消息框：info / warn / confirm。"""

    ICONS = {"info": "ℹ️", "warn": "⚠️", "question": "❓"}

    def __init__(self, parent, title, text, kind="info", rich_text=False):
        super().__init__(title, parent, width=440)
        self._result = "ok"
        row = QHBoxLayout()
        row.setSpacing(10)
        icon = QLabel(self.ICONS.get(kind, "ℹ️"))
        icon.setStyleSheet("font-size: 26px; background: transparent;")
        row.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)
        self.lbl = QLabel(text)
        self.lbl.setObjectName("dlgText")
        self.lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl.setWordWrap(True)
        row.addWidget(self.lbl, 1)
        self.body_lay.addLayout(row)
        self.body_lay.addStretch(1)

        btns = QHBoxLayout()
        btns.setSpacing(8)
        btns.addStretch(1)
        if kind == "question":
            self.btn_no = QPushButton("取消")
            self.btn_no.setObjectName("dlgBtnCancel")
            self.btn_no.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.btn_no.clicked.connect(self._on_no)
            btns.addWidget(self.btn_no)
            self.btn_yes = QPushButton("确定")
            self.btn_yes.setObjectName("dlgBtnOK")
            self.btn_yes.setDefault(True)
            self.btn_yes.clicked.connect(self._on_yes)
            btns.addWidget(self.btn_yes)
            self._result = "no"
        else:
            self.btn_ok = QPushButton("确定")
            self.btn_ok.setObjectName("dlgBtnOK")
            self.btn_ok.setDefault(True)
            self.btn_ok.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.btn_ok.clicked.connect(self.accept)
            btns.addWidget(self.btn_ok)
        self.body_lay.addLayout(btns)
        self._center_on(parent)

    def _on_yes(self):
        self._result = "yes"
        self.accept()

    def _on_no(self):
        self._result = "no"
        self.reject()

    @staticmethod
    def info(parent, title, text):
        d = NfMessage(parent, title, text, "info")
        d.exec()

    @staticmethod
    def warn(parent, title, text):
        d = NfMessage(parent, title, text, "warn")
        d.exec()

    @staticmethod
    def confirm(parent, title, text, yes_text="确定", no_text="取消") -> bool:
        d = NfMessage(parent, title, text, "question")
        d.btn_yes.setText(yes_text)
        d.btn_no.setText(no_text)
        d.exec()
        return d._result == "yes"


class NfInput(NfDialog):
    """无边框单行输入框（替代 QInputDialog.getText）。"""

    def __init__(self, parent, title, label, default="", width=440):
        super().__init__(title, parent, width=width)
        self.lbl = QLabel(label)
        self.lbl.setObjectName("dlgText")
        self.body_lay.addWidget(self.lbl)
        self.edit = QLineEdit(default)
        self.body_lay.addWidget(self.edit)
        self.body_lay.addStretch(1)
        btns = QHBoxLayout()
        btns.setSpacing(8)
        btns.addStretch(1)
        b_cancel = QPushButton("取消")
        b_cancel.setObjectName("dlgBtnCancel")
        b_cancel.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b_cancel.clicked.connect(self.reject)
        btns.addWidget(b_cancel)
        b_ok = QPushButton("确定")
        b_ok.setObjectName("dlgBtnOK")
        b_ok.setDefault(True)
        b_ok.clicked.connect(self.accept)
        btns.addWidget(b_ok)
        self.body_lay.addLayout(btns)
        self.edit.setFocus()
        self._center_on(parent)

    @staticmethod
    def get_text(parent, title, label, default="", width=440):
        d = NfInput(parent, title, label, default, width=width)
        if d.exec():
            return d.edit.text().strip(), True
        return "", False


# ---------------------------------------------------------------------------
# 9. 配置对话框
# ---------------------------------------------------------------------------

class ConfigDialog(NfDialog):
    """配置输入对话框（2026-09-07：无边框圆角卡片，与主窗风格统一）。"""

    def __init__(self, config: ConfigManager, parent=None):
        super().__init__("配置", parent, width=580)
        self.config = config
        self._build_ui()
        self._load_values()
        self._center_on(parent)

    def _build_ui(self):
        layout = self.body_lay  # 内容装入无边框卡片 body
        form = QFormLayout()

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("sk-...")
        form.addRow("DeepSeek API Key：", self.api_key_edit)

        self.base_url_edit = QLineEdit()
        form.addRow("API 地址：", self.base_url_edit)

        self.model_edit = QLineEdit()
        form.addRow("模型名称：", self.model_edit)

        self.model_dir_edit = QLineEdit()
        form.addRow("模型目录：", self.model_dir_edit)

        # 语音识别模型选择（延迟 import，避免启动卡顿）
        from asr_engine import ASR_MODELS, DEFAULT_ASR_MODEL
        self.asr_model_combo = QComboBox()
        # 默认模型(SenseVoice)置顶
        self._asr_model_keys = ([DEFAULT_ASR_MODEL] +
                                [k for k in ASR_MODELS if k != DEFAULT_ASR_MODEL])
        for k in self._asr_model_keys:
            self.asr_model_combo.addItem(ASR_MODELS[k]["label"], k)
        self.asr_model_combo.setToolTip(
            "识别引擎：SenseVoice（CPU 首选·口语稳）为默认；\n"
            "Fun-ASR-Nano 需 GPU/强 CPU（规范长文本质量最高）；\n"
            "Paraformer 中文+字级时间戳、CPU 较快。\n"
            "切换后需重启应用生效（首次加载新模型需下载对应文件）。"
        )
        form.addRow("语音识别模型：", self.asr_model_combo)

        # 模型说明（随选择更新）
        self.asr_model_desc = QLabel()
        self.asr_model_desc.setObjectName("asrModelDesc")   # 样式见 theme.py
        self.asr_model_desc.setWordWrap(True)
        self.asr_model_combo.currentIndexChanged.connect(self._on_asr_model_changed)
        form.addRow("", self.asr_model_desc)

        # VAD 切句灵敏度（延迟 import，避免启动卡顿）
        from asr_engine import VAD_PRESETS, DEFAULT_VAD_LEVEL
        self.vad_combo = QComboBox()
        self._vad_keys = list(VAD_PRESETS.keys())
        for k in self._vad_keys:
            self.vad_combo.addItem(VAD_PRESETS[k]["label"], k)
        self.vad_combo.setToolTip(
            "切句灵敏度：停顿多久算一句话结束。\n"
            "细：停顿 0.8s 即切（句子多、短促）\n"
            "中：停顿 1.5s 才切（推荐，会议平衡）\n"
            "粗：停顿 2.5s 才切（长句完整，可能混说话人）\n"
            "改后保存即生效（自动重新加载模型）。"
        )
        form.addRow("切句灵敏度：", self.vad_combo)

        # 说话人数量（2026-09-07 路线C：谱聚类后端支持 1..15，3 人+ 会议建议手动预设）
        self.spk_combo = QComboBox()
        self._spk_options = [("自动估计（1~2 人）", 0), ("2 人", 2), ("3 人", 3),
                             ("4 人", 4), ("5 人", 5), ("6 人", 6), ("8 人", 8)]
        for _label, _val in self._spk_options:
            self.spk_combo.addItem(_label, _val)
        self.spk_combo.setToolTip(
            "说话人数量：默认自动估计，适合 1~2 人会议。\n"
            "3 人及以上会议建议手动指定人数，说话人聚类将按该数量拆分。\n"
            "（需 ≥20 个语音段才会聚类；单人音频自动归为 1 人）"
        )
        form.addRow("说话人数量：", self.spk_combo)

        # 界面主题（阶段 E：深色蓝紫 / 浅色）
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("深色（蓝紫）", "dark")
        self.theme_combo.addItem("浅色（新拟态）", "light")
        self.theme_combo.setToolTip("切换后立即生效（浅色为新拟态柔和风格）。")
        form.addRow("界面主题：", self.theme_combo)

        # 总结模板（2026-09-07：工具栏可即时切换，这里可同步设置）
        self.summary_tpl_combo = QComboBox()
        for _k, (_label, _) in SUMMARY_TEMPLATES.items():
            self.summary_tpl_combo.addItem(_label, _k)
        self.summary_tpl_combo.setToolTip(
            "选择 AI 总结模板。也可在工具栏即时切换。\n"
            "会议纪要：标准结构（会议/决策/行动/遗留）\n"
            "通话摘要：双方案要+共识+后续约定\n"
            "任务待办：自动提炼所有待办清单\n"
            "知识要点：讲座/学习等转录的要点提炼"
        )
        form.addRow("总结模板：", self.summary_tpl_combo)

        layout.addLayout(form)

        tip = QLabel("提示：API Key 仅保存在本地 config.json，不会上传。")
        tip.setObjectName("tipLabel")                       # 样式见 theme.py
        layout.addWidget(tip)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        _b_ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        _b_ok.setText("保存")
        _b_ok.setObjectName("dlgBtnOK")
        _b_cancel = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        _b_cancel.setText("取消")
        _b_cancel.setObjectName("dlgBtnCancel")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_asr_model_changed(self, idx: int):
        from asr_engine import ASR_MODELS
        if 0 <= idx < len(self._asr_model_keys):
            self.asr_model_desc.setText(ASR_MODELS[self._asr_model_keys[idx]]["desc"])

    def _load_values(self):
        from asr_engine import DEFAULT_ASR_MODEL as _DEFAULT_MODEL
        self.api_key_edit.setText(self.config.get("deepseek_api_key", ""))
        self.base_url_edit.setText(self.config.get("deepseek_base_url", DEFAULT_CONFIG["deepseek_base_url"]))
        self.model_edit.setText(self.config.get("deepseek_model", DEFAULT_CONFIG["deepseek_model"]))
        self.model_dir_edit.setText(self.config.get("model_dir", DEFAULT_CONFIG["model_dir"]))
        # 恢复已保存的识别模型选择
        cur = self.config.get("asr_model", _DEFAULT_MODEL)
        idx = self._asr_model_keys.index(cur) if cur in self._asr_model_keys else \
            (self._asr_model_keys.index(_DEFAULT_MODEL) if _DEFAULT_MODEL in self._asr_model_keys else 0)
        self.asr_model_combo.setCurrentIndex(idx)
        self._on_asr_model_changed(idx)
        # 恢复已保存的切句灵敏度
        cur_vad = self.config.get("vad_level", "")
        idx_vad = self._vad_keys.index(cur_vad) if cur_vad in self._vad_keys else 0
        self.vad_combo.setCurrentIndex(idx_vad)
        # 恢复说话人数量
        _n = int(self.config.get("num_speakers", 0) or 0)
        _idx_spk = self.spk_combo.findData(_n)
        self.spk_combo.setCurrentIndex(_idx_spk if _idx_spk >= 0 else 0)
        # 恢复主题选择
        idx_theme = self.theme_combo.findData(self.config.get("theme", "light"))
        self.theme_combo.setCurrentIndex(idx_theme if idx_theme >= 0 else 0)
        # 恢复总结模板
        _idx_tpl = self.summary_tpl_combo.findData(
            self.config.get("summary_template", DEFAULT_SUMMARY_TEMPLATE))
        self.summary_tpl_combo.setCurrentIndex(_idx_tpl if _idx_tpl >= 0 else 0)

    def _on_accept(self):
        key = self.api_key_edit.text().strip()
        if not key:
            NfMessage.warn(self, "提示", "API Key 不能为空。")
            return
        self.config.set("deepseek_api_key", key)
        self.config.set("deepseek_base_url", self.base_url_edit.text().strip())
        self.config.set("deepseek_model", self.model_edit.text().strip())
        self.config.set("model_dir", self.model_dir_edit.text().strip() or "mod")
        cur = self.asr_model_combo.currentData()
        if cur:
            self.config.set("asr_model", cur)
        cur_vad = self.vad_combo.currentData()
        if cur_vad:
            self.config.set("vad_level", cur_vad)
        self.config.set("num_speakers", int(self.spk_combo.currentData() or 0))
        cur_theme = self.theme_combo.currentData()
        if cur_theme:
            self.config.set("theme", cur_theme)
        self.config.set("summary_template", self.summary_tpl_combo.currentData() or DEFAULT_SUMMARY_TEMPLATE)
        self.config.save()
        logger.info(f"配置已更新（asr_model={cur}, vad_level={cur_vad}, theme={cur_theme}）")
        self.accept()


class SeekSlider(QSlider):
    """点击轨道直接跳转的进度条。

    QSS（QStyleSheetStyle）接管后，QSlider 原生"点击轨道跳转"行为会退化失效
    （2026-09-07 真机实测：仅能拖动 thumb，点轨道无反应）。本子类把左键点击
    groove 换算成对应 value 并发出 seekRequested；点在 thumb 上仍交原生拖动；
    点击后按住不放继续移动同样换算跳转（对齐主流播放器交互）。
    """

    seekRequested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._drag_from_groove = False

    # ---- 几何换算 ----
    def _sub_rects(self):
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt,
            QStyle.SubControl.SC_SliderGroove, self)
        handle = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt,
            QStyle.SubControl.SC_SliderHandle, self)
        return groove, handle

    def _value_at_x(self, x: float):
        groove, handle = self._sub_rects()
        span = groove.width() - handle.width()
        if span <= 0 or self.maximum() <= self.minimum():
            return None
        # 点击 x 对齐到 handle 中心，再线性换算成 value（sliderValueFromPosition 需 int）
        xc = int(round(x - groove.x() - handle.width() / 2.0))
        return QStyle.sliderValueFromPosition(
            self.minimum(), self.maximum(), xc, span)

    def _jump_to_x(self, x: float):
        v = self._value_at_x(x)
        if v is None:
            return
        v = max(self.minimum(), min(self.maximum(), v))
        if v != self.value():
            self.setValue(v)
        self.seekRequested.emit(int(v))

    # ---- 鼠标事件 ----
    def _px(self, e) -> tuple:
        if hasattr(e, "position"):      # Qt6
            return float(e.position().x()), float(e.position().y())
        return float(e.x()), float(e.y())

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.maximum() > self.minimum():
            x, y = self._px(e)
            _, handle = self._sub_rects()
            if not handle.contains(int(x), int(y)):
                self._drag_from_groove = True
                self._jump_to_x(x)
                return  # 吞掉本次按下，避免 QSS 下二次 pageStep 跳变
        self._drag_from_groove = False
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._drag_from_groove and (e.buttons() & Qt.MouseButton.LeftButton):
            self._jump_to_x(self._px(e)[0])
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._drag_from_groove = False
        super().mouseReleaseEvent(e)


# ---------------------------------------------------------------------------
# 10. 主窗口
# ---------------------------------------------------------------------------

class _PreloadSignal(QObject):
    """后台预加载线程 -> UI 主线程的信号桥。"""
    progress = pyqtSignal(str, str)          # (step, detail)
    state_changed = pyqtSignal(str, str)     # (state, error)

    def emit_progress(self, step: str, detail: str = ""):
        self.progress.emit(step, detail)

    def emit_state(self, state: str, error: str = ""):
        self.state_changed.emit(state, error)


class MainWindow(QMainWindow):
    """应用主窗口。"""

    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config
        self.current_audio = ""
        self.current_audio_name = ""
        self.current_transcript = ""
        self.current_record_id = None   # 当前选中的历史记录 id（阶段 B）
        self._pending_result_obj = None
        self._current_result_obj = None
        self.recorder = None
        self.transcriber = None
        self.summarizer = None
        self.recording_timer = None
        self.record_start_time = 0.0
        self._model_manager = None
        self._preload_sig = None
        self._preload_poll_timer = None
        # 阶段 E：当前主题配色（富文本/高亮用），由 _apply_theme 更新
        self._palette = get_colors(self.config.get("theme", "light"))

        # ---- 阶段 C：播放器状态 ----
        self.player = None
        self._play_source_path = ""   # 当前播放源（记录音频 wav）
        self._play_highlighter_timer = None
        self._play_current_idx = -1    # 当前播放到的句子索引
        self._sentences_meta = []      # 当前渲染的句子元数据 [{start_ms,end_ms,start_html_idx}...]
        self._sentences_start_pos = []  # 每句在 QTextBrowser 文档中的 char 位置（用于高亮）
        self._play_state_last = "stopped"  # 状态轮询缓存（替代 playbackStateChanged 信号）
        self._play_stop_ticks = 0          # 连续 tick 报 stopped 计数（防加载期误判）
        self._pending_seek_ms = None       # 媒体未就绪时的时间戳跳转缓存（加载完成后执行）
        self._seek_guard_until = 0.0       # 刚 seek 后的防回刷截止时刻（monotonic）
        self._drag_off = None              # 无边框窗口拖动偏移（按下时记录）
        # 空格键 = 播放/暂停（应用级事件过滤器，见 eventFilter；文本输入/浏览控件放行）
        app_inst = QApplication.instance()
        if app_inst is not None:
            app_inst.installEventFilter(self)
        if QT_MULTIMEDIA_OK:
            try:
                from player import AudioPlayer
                self.player = AudioPlayer(self)
                self.player.positionChanged.connect(self._on_play_position)
                self.player.durationChanged.connect(self._on_play_duration)
                # 注意：不连 player 的 playbackStateChanged（PyQt6 6.11 连了会闪退），
                # 状态改用高亮定时器轮询 playback_state()（见 _on_play_tick）
                self.player.errorOccurred.connect(self._on_play_error)
                # 阶段 F：真机验证诊断（后端 / 插件路径 / 音频输出设备枚举）
                self._log_play_diagnostics()
            except Exception as e:
                logger.warning(f"播放器初始化失败: {e}")
                self.player = None

        # 历史记录存储（阶段 B）
        # 注意：RecordStore 根目录传 APP_DIR，内部会自动拼 records/ 子目录
        try:
            from store import RecordStore
            self.store = RecordStore(str(APP_DIR))
            # 防御性：扫描旧 recordings/*.wav 转成记录（幂等，迁移后不删源）
            legacy_dir = APP_DIR / "recordings"
            if legacy_dir.exists():
                n = RecordStore.migrate_legacy(self.store, str(legacy_dir))
                if n:
                    logger.info(f"旧录音迁移: {n} 条")
        except Exception as e:
            logger.warning(f"历史记录存储初始化失败: {e}")
            self.store = None

        self._build_ui()
        self._apply_theme()
        self._refresh_hist_list()   # 阶段 B：加载历史列表
        self._check_first_run()
        # 注意：不再启动时自动加载模型，由用户点击「🧠 加载模型」手动加载
        self._update_load_btn_state()

    # ---------- 启动预加载模型 ----------
    def _log_play_diagnostics(self):
        """阶段 F：把媒体后端、Qt 插件路径、音频输出设备写入日志（真机验证依据）。"""
        try:
            from PyQt6.QtMultimedia import QMediaDevices
            from PyQt6.QtCore import QLibraryInfo
            backend = os.environ.get("QT_MEDIA_BACKEND", "(自动选择)")
            plugin_path = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
            mm_dir = Path(plugin_path) / "multimedia"
            plugins = sorted(p.name for p in mm_dir.glob("*.dll")) if mm_dir.exists() else []
            devs = [d.description() for d in QMediaDevices.audioOutputs()]
            logger.info(
                f"[播放诊断] backend={backend} | 插件路径={plugin_path} "
                f"| multimedia 插件={plugins or '缺失!'} | 音频输出设备={devs or '未枚举到!'}"
            )
        except Exception as e:
            logger.warning(f"[播放诊断] 输出失败: {e}")

    def _on_play_error(self, msg: str):
        """播放出错：状态栏 + 日志（player 已记录详细日志）。"""
        logger.error(f"播放错误回调: {msg}")
        self.lbl_status.setText(f"播放出错：{msg}")

    def _update_load_btn_state(self):
        """根据模型加载状态更新「加载模型」按钮的文案/可用性。"""
        if not hasattr(self, "btn_load_model"):
            return
        try:
            import model_manager as _mm
            mgr = self._model_manager or _mm._default_manager
        except Exception:
            mgr = None
        if mgr is not None:
            st = mgr.state
            if st == "ready":
                self.btn_load_model.setText("✅ 模型已就绪")
                self.btn_load_model.setEnabled(False)
                return
            if st == "loading":
                self.btn_load_model.setText("⏳ 加载中…")
                self.btn_load_model.setEnabled(False)
                return
        # idle / failed / 无管理器：可点击
        self.btn_load_model.setText("🧠 加载模型")
        self.btn_load_model.setEnabled(True)

    def _on_load_model_clicked(self):
        """手动加载模型（替代启动自动预加载）。"""
        logger.info("[按钮] 点击「加载模型」")
        status = self._check_model_status()
        if status.startswith("⚠️"):
            NfMessage.warn(
                self, "模型未就绪",
                f"{status}\n\n请先在「⚙️ 配置」中检查模型目录，或确认模型文件完整。",
            )
            return
        self.btn_load_model.setText("⏳ 加载中…")
        self.btn_load_model.setEnabled(False)
        self._start_model_preload()

    def _start_model_preload(self):
        """后台加载 ASR 模型（供手动「加载模型」按钮与配置变更后调用）。

        仅当模型目录就绪时加载；否则保持 idle。
        进度通过定时轮询状态更新到状态栏（后台线程不能直接碰 UI）。
        """
        model_dir = self.config.get("model_dir", "mod")
        mp = Path(model_dir)
        if not mp.is_absolute():
            mp = APP_DIR / model_dir
        # 按配置的识别模型检查对应主模型文件是否就绪
        try:
            from asr_engine import ASR_MODELS, DEFAULT_ASR_MODEL
        except Exception:
            return
        model_key = self.config.get("asr_model", DEFAULT_ASR_MODEL)
        sub = model_key if model_key in ASR_MODELS else DEFAULT_ASR_MODEL
        if not (mp / sub).exists():
            logger.warning(f"模型目录未就绪（{sub}），无法加载")
            self.lbl_status.setText(f"⚠️ 模型目录未就绪：{mp / sub}")
            self._update_load_btn_state()
            return

        try:
            from model_manager import get_manager
            vad_level = self.config.get("vad_level", "medium")
            manager = get_manager(model_root=str(mp), asr_model=sub,
                                  vad_level=vad_level)
        except Exception as e:
            logger.warning(f"ModelManager 初始化失败: {e}")
            self.lbl_status.setText(f"⚠️ 模型管理器初始化失败: {e}")
            self._update_load_btn_state()
            return

        # 若已在加载或已就绪，则只需同步按钮状态
        if manager.state in ("loading", "ready"):
            self._update_load_btn_state()
            if manager.state == "ready":
                self.lbl_status.setText("✅ 模型已就绪，可直接转写")
            return

        self._model_manager = manager
        self.lbl_status.setText("正在加载模型（首次约需 1 分钟，可先进行其他操作）…")

        # 进度回调：后台线程 -> 信号 -> 主线程
        if self._preload_sig is None:
            self._preload_sig = _PreloadSignal()
            self._preload_sig.progress.connect(self._on_preload_progress)
            self._preload_sig.state_changed.connect(self._on_preload_state)

        manager.load_async(progress_callback=self._preload_sig.emit_progress)

        # 主线程 QTimer 轮询状态，后台加载完成/失败时刷新状态栏与按钮
        from PyQt6.QtCore import QTimer
        if self._preload_poll_timer is not None:
            self._preload_poll_timer.stop()
        self._preload_poll_count = 0
        def _poll():
            if manager is None:
                return
            st = manager.state
            if st in ("ready", "failed"):
                self._on_preload_state(st, manager.error)
                self._preload_poll_timer.stop()
                self._update_load_btn_state()
            else:
                self._preload_poll_count += 1
                if self._preload_poll_count % 4 == 0:  # ~每 2s 刷新一次提示
                    self._on_preload_state(st)
        self._preload_poll_timer = QTimer(self)
        self._preload_poll_timer.timeout.connect(_poll)
        self._preload_poll_timer.start(500)

    def _on_preload_progress(self, step: str, detail: str = ""):
        text = f"{step}：{detail}" if detail else step
        logger.debug(f"预加载进度: {text}")
        self.lbl_status.setText(f"⏳ {text}")

    def _on_preload_state(self, state: str, error: str = ""):
        logger.info(f"预加载状态: {state}")
        if state == "ready":
            self.lbl_status.setText("✅ 模型已就绪，可直接转写")
        elif state == "failed":
            self.lbl_status.setText(f"⚠️ 模型加载失败（转写时会重试）: {error[:60]}")
        elif state == "loading":
            self.lbl_status.setText("正在后台加载模型（可先进行其他操作）…")

    # ---------- UI 构建 ----------
    def _build_ui(self):
        # 无边框 + 透明背景：整窗圆角由 QWidget#appRoot 承载（2026-09-07 用户需求：去顶部标题、整窗圆角）
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # 窗口标题保留给任务栏 / Alt+Tab 识别（无边框后不可见，不占用界面）
        self.setWindowTitle("智能会议纪要工具")
        self.resize(1100, 760)

        central = QWidget()
        central.setObjectName("appRoot")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 0, 14, 14)
        root.setSpacing(6)

        # 无边框窗口标题栏：无文字，仅拖动区 + 最小化/最大化/关闭（2026-09-07 去顶部大字）
        self.title_bar = QWidget()
        self.title_bar.setObjectName("titleBar")
        self.title_bar.setFixedHeight(30)
        tb_lay = QHBoxLayout(self.title_bar)
        tb_lay.setContentsMargins(10, 0, 6, 0)
        tb_lay.setSpacing(2)
        tb_lay.addStretch(1)
        self.btn_win_min = QPushButton("─")
        self.btn_win_min.setObjectName("btnWinMin")
        self.btn_win_min.setFixedSize(34, 24)
        self.btn_win_min.setToolTip("最小化")
        self.btn_win_min.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_win_min.clicked.connect(self.showMinimized)
        tb_lay.addWidget(self.btn_win_min)
        self.btn_win_max = QPushButton("□")
        self.btn_win_max.setObjectName("btnWinMax")
        self.btn_win_max.setFixedSize(34, 24)
        self.btn_win_max.setToolTip("最大化/还原")
        self.btn_win_max.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_win_max.clicked.connect(self._toggle_max)
        tb_lay.addWidget(self.btn_win_max)
        self.btn_win_close = QPushButton("✕")
        self.btn_win_close.setObjectName("btnWinClose")
        self.btn_win_close.setFixedSize(38, 24)
        self.btn_win_close.setToolTip("关闭")
        self.btn_win_close.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_win_close.clicked.connect(self.close)
        tb_lay.addWidget(self.btn_win_close)
        # 标题栏拖动 / 双击最大化
        self.title_bar.mousePressEvent = self._titlebar_press
        self.title_bar.mouseMoveEvent = self._titlebar_move
        self.title_bar.mouseReleaseEvent = self._titlebar_release
        self.title_bar.mouseDoubleClickEvent = self._titlebar_dblclick
        root.addWidget(self.title_bar)

        # 工具栏
        toolbar = QHBoxLayout()
        self.btn_record = QPushButton("🎤 开始录音")
        self.btn_record.setObjectName("btnRecord")
        self.btn_stop = QPushButton("⏹ 停止录音")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setEnabled(False)
        self.btn_import = QPushButton("📂 导入文件")
        self.btn_import.setObjectName("btnImport")
        toolbar.addWidget(self.btn_record)
        toolbar.addWidget(self.btn_stop)
        toolbar.addWidget(self.btn_import)
        toolbar.addStretch(1)
        self.btn_load_model = QPushButton("🧠 加载模型")
        self.btn_load_model.setObjectName("btnLoadModel")
        self.btn_load_model.setToolTip("手动加载语音识别模型（首次约需 1 分钟；加载后转写无需再等待）")
        self.btn_load_model.clicked.connect(self._on_load_model_clicked)
        toolbar.addWidget(self.btn_load_model)
        self.btn_config = QPushButton("⚙️ 配置")
        self.btn_config.setObjectName("btnConfig")
        toolbar.addWidget(self.btn_config)
        root.addLayout(toolbar)

        # 音频信息条
        audio_bar = QHBoxLayout()
        self.lbl_audio = QLabel("音频文件：无")
        self.lbl_audio.setObjectName("lblAudio")
        self.lbl_duration = QLabel("时长：--")
        self.lbl_duration.setObjectName("lblDuration")
        audio_bar.addWidget(self.lbl_audio)
        audio_bar.addSpacing(20)
        audio_bar.addWidget(self.lbl_duration)
        audio_bar.addStretch(1)
        root.addLayout(audio_bar)

        # 分栏：转写 + 纪要
        splitter = QSplitter(Qt.Orientation.Vertical)

        # 转写区
        tg = QWidget()
        tl = QVBoxLayout(tg)
        tl.setContentsMargins(0, 0, 0, 0)
        th = QHBoxLayout()
        th.addWidget(QLabel("📝 转写结果（含说话人）"))
        lbl_ts_tip = QLabel("点时间戳跳转 · 点说话人改名/合并")
        lbl_ts_tip.setObjectName("lblTranscriptTip")
        th.addWidget(lbl_ts_tip)
        th.addStretch(1)
        self.btn_transcribe = QPushButton("🔊 开始转写")
        self.btn_transcribe.setObjectName("btnTranscribe")
        th.addWidget(self.btn_transcribe)
        tl.addLayout(th)
        self.txt_transcript = QTextBrowser()
        self.txt_transcript.setObjectName("txtTranscript")
        self.txt_transcript.setOpenLinks(False)  # 捕获链接点击自己做跳转
        self.txt_transcript.setPlaceholderText("转写结果将显示在这里（带时间戳和说话人标签）…")
        # 阶段 C：点击句子时间戳链接 -> 跳转播放
        self.txt_transcript.anchorClicked.connect(self._on_ts_link_clicked)
        tl.addWidget(self.txt_transcript)

        # 纪要区
        sg = QWidget()
        sl = QVBoxLayout(sg)
        sl.setContentsMargins(0, 0, 0, 0)
        sh = QHBoxLayout()
        sh.addWidget(QLabel("📋 会议纪要"))
        sh.addStretch(1)
        # 总结模板下拉（2026-09-07：会议/通话/待办/要点四套）
        self.cmb_summary_tpl = QComboBox()
        self.cmb_summary_tpl.setObjectName("cmbSummaryTpl")
        for _k, (_label, _) in SUMMARY_TEMPLATES.items():
            self.cmb_summary_tpl.addItem(_label, _k)
        self.cmb_summary_tpl.setToolTip("选择 AI 总结模板。可随时切换，下次生成时生效。")
        self.cmb_summary_tpl.setCurrentIndex(
            self.cmb_summary_tpl.findData(self.config.get("summary_template", DEFAULT_SUMMARY_TEMPLATE))
        )
        self.cmb_summary_tpl.currentIndexChanged.connect(self._on_summary_tpl_changed)
        sh.addWidget(self.cmb_summary_tpl)
        self.btn_summarize = QPushButton("✨ 生成纪要")
        self.btn_summarize.setObjectName("btnSummarize")
        self.btn_export = QPushButton("💾 导出纪要")
        self.btn_export.setObjectName("btnExport")
        sh.addWidget(self.btn_summarize)
        sh.addWidget(self.btn_export)
        sl.addLayout(sh)
        self.txt_summary = QPlainTextEdit()
        self.txt_summary.setObjectName("txtSummary")
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setPlaceholderText("会议纪要将显示在这里…")
        sl.addWidget(self.txt_summary)

        splitter.addWidget(tg)
        splitter.addWidget(sg)
        splitter.setSizes([400, 300])

        # ---- 阶段 B：左右分栏 = 历史栏 + 原内容 ----
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setObjectName("mainSplitter")
        self.main_splitter.setChildrenCollapsible(False)

        # 左：历史记录栏
        hist_panel = QWidget()
        hist_panel.setObjectName("histPanel")
        hist_panel.setMinimumWidth(190)
        hist_panel.setMaximumWidth(320)
        hl = QVBoxLayout(hist_panel)
        hl.setContentsMargins(6, 6, 6, 6)
        hl.setSpacing(4)
        hist_title = QLabel("🗂 历史记录")
        hist_title.setObjectName("histTitle")
        hl.addWidget(hist_title)
        self.hist_list = QListWidget()
        self.hist_list.setObjectName("histList")
        self.hist_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.hist_list.customContextMenuRequested.connect(self._on_hist_menu)
        self.hist_list.itemClicked.connect(self._on_hist_selected)
        self.hist_list.setToolTip("右键可重命名 / 删除记录")
        hl.addWidget(self.hist_list, stretch=1)
        btn_row = QHBoxLayout()
        self.btn_hist_new = QPushButton("＋ 新记录")
        self.btn_hist_new.setObjectName("btnHistNew")
        self.btn_hist_new.setToolTip("从当前音频新建一条记录（用于手动归档）")
        self.btn_hist_new.clicked.connect(self._on_hist_new)
        btn_row.addWidget(self.btn_hist_new)
        hl.addLayout(btn_row)
        self.main_splitter.addWidget(hist_panel)

        # 右：原内容容器（把 splitter 塞进一个容器 widget）
        right_wrap = QWidget()
        rl = QVBoxLayout(right_wrap)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(splitter)
        self.main_splitter.addWidget(right_wrap)
        self.main_splitter.setSizes([210, 890])
        root.addWidget(self.main_splitter, stretch=1)

        # ---- 阶段 C：底部播放条 ----
        if QT_MULTIMEDIA_OK:
            play_bar = QWidget()
            play_bar.setObjectName("playBar")
            pbl = QHBoxLayout(play_bar)
            pbl.setContentsMargins(4, 2, 4, 2)
            pbl.setSpacing(6)
            self.btn_play = QPushButton("▶ 播放")
            self.btn_play.setObjectName("btnPlay")
            self.btn_play.setEnabled(False)
            self.btn_play.setToolTip("播放当前音频（点击转写中的时间戳可跳转）")
            self.btn_play.clicked.connect(self._on_play_clicked)
            pbl.addWidget(self.btn_play)
            self.btn_play_stop = QPushButton("⏹")
            self.btn_play_stop.setObjectName("btnPlayStop")
            self.btn_play_stop.setEnabled(False)
            self.btn_play_stop.setToolTip("停止播放")
            self.btn_play_stop.clicked.connect(self._on_play_stop_clicked)
            pbl.addWidget(self.btn_play_stop)
            self.slider_pos = SeekSlider()
            self.slider_pos.setObjectName("sliderPos")
            self.slider_pos.setEnabled(False)
            self.slider_pos.setRange(0, 0)
            self.slider_pos.setToolTip("点击或拖动跳转播放位置")
            # 拖动 thumb / 点击轨道（seekRequested）都走同一跳转处理
            self.slider_pos.sliderMoved.connect(self._on_slider_moved)
            self.slider_pos.seekRequested.connect(self._on_slider_moved)
            pbl.addWidget(self.slider_pos, stretch=1)
            self.lbl_play_time = QLabel("00:00 / 00:00")
            self.lbl_play_time.setObjectName("lblPlayTime")
            self.lbl_play_time.setMinimumWidth(140)
            pbl.addWidget(self.lbl_play_time)
            root.addWidget(play_bar)
        else:
            # QtMultimedia 不可用时隐藏播放条（仍保留 btn_play 引用避免异常）
            self.btn_play = None
            self.btn_play_stop = None
            self.slider_pos = None
            self.lbl_play_time = None

        # 进度条
        self.progress = QProgressBar()
        self.progress.setObjectName("progressBar")
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        # 状态栏：普通圆角容器（替代 QStatusBar，配合无边框+圆角布局，2026-09-07）
        self.status_bar = QWidget()
        self.status_bar.setObjectName("statusBar")
        self.status_bar.setFixedHeight(30)
        sb_lay = QHBoxLayout(self.status_bar)
        sb_lay.setContentsMargins(10, 3, 10, 3)
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setObjectName("lblStatus")
        sb_lay.addWidget(self.lbl_status)
        sb_lay.addStretch(1)
        root.addWidget(self.status_bar)

        # 信号连接
        self.btn_record.clicked.connect(self.on_record_clicked)
        self.btn_stop.clicked.connect(self.on_stop_clicked)
        self.btn_import.clicked.connect(self.on_import_clicked)
        self.btn_transcribe.clicked.connect(self.on_transcribe_clicked)
        self.btn_summarize.clicked.connect(self.on_summarize_clicked)
        self.btn_export.clicked.connect(self.on_export_clicked)
        self.btn_config.clicked.connect(self.on_config_clicked)

    # ---------- 首次运行 ----------
    def _check_model_status(self) -> str:
        """返回模型状态文本：就绪的模型名 / 未就绪原因。

        按配置中的 asr_model 检查对应主模型是否存在（VAD/说话人公共组件
        由引擎加载时统一校验）。返回的字符串供状态栏显示。
        """
        try:
            from asr_engine import ASR_MODELS
        except Exception:
            return "⚠️ 模型模块不可用"
        model_key = self.config.get("asr_model", "sensevoice")
        sub = model_key if model_key in ASR_MODELS else "sensevoice"
        model_dir = self.config.get("model_dir", "mod")
        mp = Path(model_dir)
        if not mp.is_absolute():
            mp = APP_DIR / model_dir
        if (mp / sub).exists():
            return f"✅ 模型已就绪（{ASR_MODELS[sub]['label']}）"
        return f"⚠️ 模型未就绪，请检查 mod/{sub} 目录"

    def _check_first_run(self):
        if not self.config.get("deepseek_api_key", ""):
            logger.info("首次运行：API Key 为空，弹出配置引导")
            QTimer.singleShot(300, self._show_first_run_dialog)
        # 检查模型目录（按配置的识别模型）
        status = self._check_model_status()
        logger.info(f"模型状态检查: {status}")
        self.lbl_status.setText(status)

    def _show_first_run_dialog(self):
        NfMessage.info(
            self, "欢迎使用",
            "欢迎使用智能会议纪要工具！\n\n"
            "首次使用请先配置 DeepSeek API Key（用于生成会议纪要）。\n"
            "语音识别模型放在程序目录的 mod/ 文件夹内。",
        )
        _old_theme = self.config.get("theme", "light")
        dlg = ConfigDialog(self.config, self)
        dlg.exec()
        # 配置弹窗可能修改了主题/识别模型：主题立即生效 + 重置并重新预加载
        self._reapply_theme_if_changed(_old_theme)
        self._restart_preload_after_config()

    def _restart_preload_after_config(self):
        """配置弹窗关闭后调用：重置管理器（不自动加载，等用户点「加载模型」）。"""
        try:
            import model_manager as _mm
            if _mm._default_manager is not None:
                _mm._default_manager.reset()
                logger.info("配置变更后 manager 已重置")
        except Exception as e:
            logger.warning(f"重置 manager 失败: {e}")
        # 清掉轮询 timer（避免旧轮询引用已重置的 manager）
        if self._preload_poll_timer is not None:
            self._preload_poll_timer.stop()
            self._preload_poll_timer = None
        self._model_manager = None
        self._update_load_btn_state()
        # 立即刷新状态栏模型状态
        self.lbl_status.setText(self._check_model_status())
        if not self._check_model_status().startswith("⚠️"):
            self.lbl_status.setText(self._check_model_status() + "（点击「🧠 加载模型」开始加载）")

    # ---------- 历史记录（阶段 B） ----------
    def _refresh_hist_list(self):
        """重建左侧历史列表（新 -> 旧）。"""
        if self.store is None:
            return
        self.hist_list.blockSignals(True)
        self.hist_list.clear()
        recs = self.store.list_records()
        recs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        for r in recs:
            rid = r.get("id", "")
            title = r.get("title", rid)
            status = r.get("status", "pending")
            icon = {"summarized": "✅", "transcribed": "📝", "pending": "🎙️"}.get(status, "📄")
            item = QListWidgetItem(f"{icon} {title}")
            item.setData(Qt.ItemDataRole.UserRole, rid)
            item.setToolTip(f"{title}\n状态: {status}\n创建: {r.get('created_at', '')}")
            self.hist_list.addItem(item)
        self.hist_list.blockSignals(False)
        if recs:
            self.lbl_status.setText(f"历史记录：{len(recs)} 条")

    def _find_hist_item(self, rid: str):
        for i in range(self.hist_list.count()):
            item = self.hist_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == rid:
                return item
        return None

    def _on_hist_selected(self, item: QListWidgetItem):
        """点击历史记录 -> 加载该记录的音频 + 转写 + 纪要。"""
        if self.store is None:
            return
        rid = item.data(Qt.ItemDataRole.UserRole)
        if not rid:
            return
        rec = self.store.get(rid)
        if rec is None:
            return
        # 若正在转写/录音，忽略切换
        if (self.transcriber is not None and self.transcriber.isRunning()) or \
           (self.recorder is not None and self.recorder.isRunning()):
            self.lbl_status.setText("请等待当前任务完成后再切换记录")
            if self.current_record_id:
                cur = self._find_hist_item(self.current_record_id)
                if cur:
                    self.hist_list.setCurrentItem(cur)
            return
        self.current_record_id = rid
        self.current_audio = str(self.store.audio_path(rid))
        self.current_audio_name = rec.get("title", rid)
        self.lbl_audio.setText(f"音频文件：{Path(self.current_audio).name}")
        dur = rec.get("duration_sec", 0)
        self.lbl_duration.setText(f"时长：{format_timestamp(dur)}" if dur else "时长：--")
        data = self.store.load_transcript(rid)
        if data and data.get("sentences"):
            try:
                from asr_engine import TranscriptionResult
                tr = TranscriptionResult.from_dict(data)
                self.current_transcript = tr.to_text()
                # 阶段 C：结构化渲染为句子列表（时间戳可点）
                header = f"{self.current_audio_name}" if self.current_audio_name else ""
                self._render_sentences(tr, header)
                self._current_result_obj = tr
            except Exception as e:
                logger.warning(f"转写解析失败: {e}")
                self._update_sentences_plain("(该记录暂无可用转写内容)")
                self.current_transcript = ""
                self._current_result_obj = None
        else:
            self.current_transcript = ""
            self._current_result_obj = None
            self._sentences_meta = []
            self.txt_transcript.setPlainText("该记录尚未转写。\n点击「🔊 开始转写」进行语音转写。")
        summary = self.store.load_summary(rid)
        if summary:
            self.txt_summary.setPlainText(summary)
        else:
            self.txt_summary.setPlainText("该记录尚未生成纪要。\n点击「✨ 生成纪要」调用 AI 总结。")
        # 阶段 C：有音频则可播放
        self._reset_play_ui()
        ap = self.store.audio_path(rid)
        if Path(ap).exists():
            if self.btn_play is not None:
                self.btn_play.setEnabled(True)
                self.btn_play_stop.setEnabled(True)
            self._play_source_path = ""  # 强制下次 set_source
        else:
            if self.btn_play is not None:
                self.btn_play.setEnabled(False)
                self.btn_play_stop.setEnabled(False)
        st = {"summarized": "✅ 已总结", "transcribed": "📝 已转写，可补做纪要",
              "pending": "🎙️ 待转写"}.get(rec.get("status", "pending"), "")
        self.lbl_status.setText(f"已打开记录：{rec.get('title', rid)}（{st}）")
        logger.info(f"打开历史记录: {rid} title={rec.get('title')} status={rec.get('status')}")

    def _on_hist_menu(self, pos):
        """历史记录右键菜单：重命名 / 删除。"""
        if self.store is None:
            return
        item = self.hist_list.itemAt(pos)
        if item is None:
            return
        rid = item.data(Qt.ItemDataRole.UserRole)
        rec = self.store.get(rid)
        if rec is None:
            return
        menu = QMenu(self)
        act_rename = menu.addAction("✏️ 重命名")
        act_delete = menu.addAction("🗑 删除记录")
        act = menu.exec(self.hist_list.viewport().mapToGlobal(pos))
        if act == act_rename:
            new_title, ok = NfInput.get_text(
                self, "重命名记录", "新名称：", default=rec.get("title", "")
            )
            if ok and new_title.strip():
                self.store.rename(rid, new_title.strip())
                self._refresh_hist_list()
                if self.current_record_id == rid:
                    self.current_audio_name = new_title.strip()
                    self.lbl_status.setText(f"已重命名为：{new_title.strip()}")
                logger.info(f"重命名记录: {rid} -> {new_title.strip()}")
        elif act == act_delete:
            if NfMessage.confirm(
                    self, "删除记录",
                    f"确定删除记录「{rec.get('title', rid)}」？\n将删除其音频、转写与纪要文件。",
                    yes_text="删除", no_text="取消"):
                was_current = (self.current_record_id == rid)
                self.store.delete(rid)
                self._refresh_hist_list()
                if was_current:
                    self._clear_current()
                self.lbl_status.setText("记录已删除")
                logger.info(f"删除记录: {rid}")

    def _clear_current(self):
        """清空当前编辑区（删除当前记录时使用）。"""
        self.current_record_id = None
        self.current_audio = ""
        self.current_audio_name = ""
        self.current_transcript = ""
        self._current_result_obj = None
        self._sentences_meta = []
        self._sentences_start_pos = []
        self._play_current_idx = -1
        self._stop_playback()
        self.lbl_audio.setText("音频文件：无")
        self.lbl_duration.setText("时长：--")
        self.txt_transcript.clear()
        self.txt_summary.clear()

    # ========== 阶段 C：转写渲染 + 播放 ==========

    def _render_sentences(self, result, header: str = ""):
        """把结构化转写结果渲染为 QTextBrowser 富文本（句子列表）。

        每句格式：可点击时间戳 + 说话人 + 文本。
        链接 href 形如 jump:<start_ms>，点击后 seek 播放。
        """
        import html as _html
        pal = getattr(self, "_palette", None) or get_colors("dark")
        meta = []
        html_parts = []
        if header:
            html_parts.append(
                f"<div style='font-weight:bold; color:{pal['accent']};'>"
                + _html.escape(header) + "</div>"
            )
        sentences = result.sentences if result else []
        for i, s in enumerate(sentences):
            text = (s.get("text") or "").strip()
            if not text:
                continue
            start_ms = int(s.get("start_ms") or 0)
            end_ms = int(s.get("end_ms") or 0)
            spk = s.get("speaker")
            if spk is not None and result and result.speakers:
                name = result.speakers.get(spk, f"说话人{spk}")
            elif spk is not None:
                name = f"说话人{spk}"
            else:
                name = ""
            ts_html = self._fmt_ts_html(start_ms, end_ms, start_ms)
            # 阶段 D：说话人标签可点击（href = spk:<编号>），点击弹菜单改名/合并
            spk_html = (
                f"<a href='spk:{spk}' style='color:{pal['spk']}; text-decoration:none;' "
                f"title='点击可重命名 / 合并说话人'>{_html.escape(name)}</a> "
            ) if (spk is not None and name) else ""
            meta.append({"start_ms": start_ms, "end_ms": end_ms, "idx": i})
            html_parts.append(
                f"<div style='margin-bottom:4px;'>{ts_html} {spk_html}"
                f"<span style='color:{pal['body']};'>{_html.escape(text)}</span></div>"
            )
        self._sentences_meta = meta
        if html_parts:
            self.txt_transcript.setHtml("".join(html_parts))
        else:
            self.txt_transcript.setPlainText("(无转写结果)")

    def _fmt_ts_html(self, start_ms: int, end_ms: int, jump_ms: int) -> str:
        """时间戳 -> 可点击 HTML（颜色跟随当前主题）。"""
        pal = getattr(self, "_palette", None) or get_colors("dark")
        ts_color = pal.get("ts", "#7aa2f7")
        s = start_ms / 1000.0
        e = end_ms / 1000.0
        h1, m1, s1 = int(s // 3600), int(s % 3600 // 60), int(s % 60)
        h2, m2, s2 = int(e // 3600), int(e % 3600 // 60), int(e % 60)
        t1 = f"{h1:02d}:{m1:02d}:{s1:02d}"
        t2 = f"{h2:02d}:{m2:02d}:{s2:02d}"
        return (
            f"<a href='jump:{int(jump_ms)}' style='color:{pal.get('ts', '#7aa2f7')}; "
            f"text-decoration:none;'>[{t1}-{t2}]</a>"
        )

    def _update_sentences_plain(self, text: str, header: str = ""):
        """无结构化结果时显示纯文本（兼容旧数据/错误信息）。"""
        self._sentences_meta = []
        if header:
            text = header + "\n" + text
        self.txt_transcript.setPlainText(text)

    def _on_ts_link_clicked(self, url: QUrl):
        """点击转写区链接：时间戳 -> 跳转播放；说话人 -> 改名/合并菜单。"""
        href = url.toString()
        if href.startswith("jump:"):
            logger.info(f"点击时间戳链接: {href}")
            try:
                ms = int(href.split(":", 1)[1])
            except Exception:
                return
            self._seek_play(ms)
        elif href.startswith("spk:"):
            logger.info(f"点击说话人标签: {href}")
            try:
                spk = int(href.split(":", 1)[1])
            except Exception:
                return
            self._on_speaker_clicked(spk)

    # ========== 阶段 D：说话人编辑（重命名 / 合并） ==========

    def _speaker_count(self, result, spk: int) -> int:
        """统计某说话人的句数。"""
        return sum(1 for s in result.sentences if s.get("speaker") == spk)

    def _is_default_speaker_name(self, name: str, spk: int) -> bool:
        """判断显示名是否还是默认的「说话人N」。"""
        return (not name) or name.strip() == f"说话人{spk}"

    def _on_speaker_clicked(self, spk: int):
        """点击说话人标签 -> 弹出菜单（重命名 / 合并到）。"""
        result = self._current_result_obj
        if result is None or not result.sentences:
            return
        cur_name = result.speakers.get(spk, f"说话人{spk}")
        others = sorted(k for k in result.speakers if k != spk)
        # 句子里出现过、但 speakers 映射缺失的编号也补进去，避免出现合并不了的目标
        for s in result.sentences:
            k = s.get("speaker")
            if k is not None and k != spk and k not in others:
                others.append(k)
        others.sort()

        menu = QMenu(self)
        act_rename = menu.addAction(f"✏️ 重命名「{cur_name}」…")
        merge_acts = {}
        if others:
            sub = menu.addMenu("🔀 合并到")
            for k in others:
                kname = result.speakers.get(k, f"说话人{k}")
                merge_acts[sub.addAction(f"{kname}（{self._speaker_count(result, k)} 句）")] = k
        act = menu.exec(QCursor.pos())
        if act is None:
            return
        if act == act_rename:
            self._rename_speaker(spk)
        elif act in merge_acts:
            self._merge_speaker(spk, merge_acts[act])

    def _rename_speaker(self, spk: int):
        """重命名某个说话人（该说话人所有句子统一更新）。"""
        result = self._current_result_obj
        if result is None:
            return
        cur_name = result.speakers.get(spk, f"说话人{spk}")
        name, ok = NfInput.get_text(
            self, "重命名说话人",
            f"为「{cur_name}」输入新名称（其全部 {self._speaker_count(result, spk)} 句将统一更新）：",
            default=cur_name,
        )
        if not ok:
            return
        name = (name or "").strip()
        if not name or name == cur_name:
            return
        result.speakers[spk] = name
        self._apply_speaker_change(f"已重命名：{cur_name} → {name}")

    def _merge_speaker(self, src: int, dst: int):
        """把说话人 src 合并到 dst（句子编号统一改为 dst）。"""
        result = self._current_result_obj
        if result is None:
            return
        src_name = result.speakers.get(src, f"说话人{src}")
        dst_name = result.speakers.get(dst, f"说话人{dst}")
        n = self._speaker_count(result, src)
        tip = f"将「{src_name}」的 {n} 句合并到「{dst_name}」，合并后统一显示为「{dst_name}」。"
        # 若源已改成真实姓名、目标仍是默认名，则保留真实姓名（更符合直觉）
        if self._is_default_speaker_name(dst_name, dst) and not self._is_default_speaker_name(src_name, src):
            tip += f"\n\n检测到「{src_name}」是自定义名称，合并后将沿用该名称。"
        tip += "\n\n此操作会改写转写结果（不可撤销，重新转写可复原）。是否继续？"
        if not NfMessage.confirm(
                self, "合并说话人", tip,
                yes_text="合并", no_text="取消"):
            return
        for s in result.sentences:
            if s.get("speaker") == src:
                s["speaker"] = dst
        # 名称取舍：保留自定义名
        if self._is_default_speaker_name(dst_name, dst) and not self._is_default_speaker_name(src_name, src):
            result.speakers[dst] = src_name
        result.speakers.pop(src, None)
        self._apply_speaker_change(f"已合并：{src_name} → {result.speakers.get(dst, f'说话人{dst}')}")

    def _apply_speaker_change(self, tip: str):
        """说话人改动后的统一收尾：重渲染 + 刷新纪要输入 + 落盘 + 提示。"""
        result = self._current_result_obj
        if result is None:
            return
        header = self.current_audio_name or ""
        self._play_current_idx = -1      # 重渲染后强制重新高亮
        self._render_sentences(result, header)
        # 纪要输入同步刷新：AI 收到的文本用新名字
        self.current_transcript = result.to_text()
        # 落盘（有记录才写；无记录时仅内存生效，等归档时一起存）
        if self.current_record_id and self.store is not None:
            try:
                ok = self.store.update_transcript(
                    self.current_record_id, result.sentences, result.speakers
                )
                logger.info(f"说话人改动已落盘: {self.current_record_id} ok={ok}")
            except Exception as e:
                logger.error(f"说话人改动落盘失败: {e}")
        self.lbl_status.setText(tip + self._summary_stale_hint())
        logger.info(f"说话人改动: {tip}")

    def _summary_stale_hint(self) -> str:
        """若已生成过纪要，提示新名字需要重新生成纪要才生效。"""
        if self.txt_summary is None:
            return ""
        txt = self.txt_summary.toPlainText().strip()
        if not txt:
            return ""
        if txt.startswith("该记录尚未生成纪要") or txt.startswith("会议纪要将显示在这里"):
            return ""
        return "（如需纪要也用新名字，请重新生成纪要）"

    def _ensure_play_source(self) -> bool:
        """确保播放源已设置（当前记录/音频的 wav）。
        返回 True 表示可播放。
        """
        if self.player is None:
            return False
        # 优先：当前历史记录的 audio.wav（与转写对齐）
        src = ""
        if self.current_record_id and self.store is not None:
            ap = self.store.audio_path(self.current_record_id)
            if Path(ap).exists():
                src = str(ap)
        if not src and self.current_audio and Path(self.current_audio).exists():
            src = self.current_audio
        if not src:
            return False
        if src != self._play_source_path:
            if not self.player.set_source(src):
                return False
            self._play_source_path = src
            # 新音源时重置进度条
            if self.slider_pos is not None:
                self.slider_pos.setRange(0, 0)
                self.slider_pos.setValue(0)
            self.lbl_play_time.setText("00:00 / 00:00")
        return True

    def _seek_play(self, ms: int):
        """跳转到指定位置播放。

        2026-09-07 修复：seek 后立即 _apply_play_position 刷新进度条/时间/高亮，
        不再等 positionChanged 信号或轮询（WMF 下 seek 异步，期间用旧位置回刷
        会造成"进度条不跟随"观感）。若媒体尚未加载完成（duration<=0，如刚导入
        音频/首次 setSource），Qt 会吞掉 seek —— 此时缓存目标位置 _pending_seek_ms，
        等 durationChanged 加载就绪后自动精确跳转。
        """
        if self.player is None or not self._ensure_play_source():
            self.lbl_status.setText("当前没有可播放的音频")
            return
        d = self.player.duration() if self.player else 0
        if d <= 0:
            # 媒体加载中：缓存待跳位置，先起播/起轮询，就绪后由 durationChanged 接手
            self._pending_seek_ms = int(ms)
            self.player.play()
            self._play_state_last = "playing"
            self._play_stop_ticks = 0
            self.lbl_status.setText(
                f"正在播放：{self.current_audio_name or '当前音频'}（{ms // 1000} 秒处）")
            self._start_highlight_timer()
            return
        self.player.seek(ms)
        self.player.play()
        self._play_state_last = "playing"
        self._play_stop_ticks = 0
        self._seek_guard_until = time.monotonic() + 0.6  # 0.6s 内轮询不回刷旧位置
        self.lbl_status.setText(f"正在播放：{self.current_audio_name or '当前音频'}（{ms // 1000} 秒处）")
        self._apply_play_position(ms)  # 立即刷新，不等 positionChanged/轮询
        self._start_highlight_timer()

    def eventFilter(self, obj, ev):
        """应用级快捷键：空格 = 播放/暂停（2026-09-07 新增）。

        仅当焦点不在文本输入/浏览控件（QLineEdit/QTextEdit/QTextBrowser 等）时拦截，
        避免抢走输入框的空格字符与转写区的空格翻页；焦点在按钮/空白处时，
        空格即播放/暂停，符合主流播放器习惯。
        """
        if (ev.type() == QEvent.Type.KeyPress and not ev.isAutoRepeat()
                and ev.key() == Qt.Key.Key_Space
                and ev.modifiers() == Qt.KeyboardModifier.NoModifier):
            fw = QApplication.focusWidget()
            if not isinstance(fw, (QLineEdit, QTextEdit, QPlainTextEdit,
                                   QTextBrowser, QComboBox, QListWidget)):
                self._on_play_clicked()
                return True
        return super().eventFilter(obj, ev)

    def _on_play_clicked(self):
        logger.info("[按钮] 点击「播放」")
        if self.player is None:
            return
        if not self._ensure_play_source():
            self.lbl_status.setText("当前没有可播放的音频（请先录音/导入并归档）")
            return
        if self.player.is_playing():
            self.player.pause()
            self._play_state_last = "paused"
            self._play_stop_ticks = 0
            self.btn_play.setText("▶ 播放")
        else:
            self.player.play()
            self._play_state_last = "playing"
            self._play_stop_ticks = 0
            self.btn_play.setText("⏸ 暂停")
            self._start_highlight_timer()

    def _on_play_stop_clicked(self):
        self._stop_playback()

    def _stop_playback(self):
        if self.player is not None:
            self.player.stop()
        self._play_state_last = "stopped"
        self._play_stop_ticks = 0
        self._pending_seek_ms = None
        self._seek_guard_until = 0.0
        if self.btn_play is not None:
            self.btn_play.setText("▶ 播放")
        if self.slider_pos is not None:
            self.slider_pos.setValue(0)
        self._play_current_idx = -1
        self._stop_highlight_timer()
        self._clear_highlight()

    def _on_slider_moved(self, pos: int):
        """拖动/点击进度条 -> 跳转（立即刷新位置/时间，不依赖 positionChanged）。"""
        if self.player is not None:
            self.player.seek(pos)
            self._seek_guard_until = time.monotonic() + 0.6  # 拖动/点击后防 tick 旧值回刷
            self._apply_play_position(pos)

    def _apply_play_position(self, ms: int):
        """统一刷新 进度条 + 时间文本 + 当前句高亮（positionChanged 与轮询 tick 共用）。"""
        if self.slider_pos is not None:
            self.slider_pos.setValue(int(ms))
        if self.lbl_play_time is not None:
            d = self.player.duration() if self.player else 0
            self.lbl_play_time.setText(
                f"{format_timestamp(ms / 1000)} / {format_timestamp(d / 1000)}"
            )
        self._highlight_current_sentence(int(ms))

    def _on_play_position(self, ms: int):
        self._apply_play_position(ms)

    def _on_play_duration(self, ms: int):
        if self.slider_pos is not None:
            self.slider_pos.setRange(0, int(ms))
            # 修复：时长就绪即启用进度条（此前从未 setEnabled(True)，无法拖动）
            self.slider_pos.setEnabled(int(ms) > 0)
        if self.lbl_play_time is not None:
            self.lbl_play_time.setText(
                f"{format_timestamp(self.player.position() / 1000)} / {format_timestamp(ms / 1000)}"
            )
        # 媒体加载完成：若有点时间戳时缓存的待跳位置（当时 duration<=0 seek 被吞），现在补跳
        if ms > 0 and self._pending_seek_ms is not None:
            pms = self._pending_seek_ms
            self._pending_seek_ms = None
            logger.info(f"[播放诊断] 媒体就绪，补执行待跳 seek: {pms} ms")
            self.player.seek(pms)
            self._play_state_last = "playing"
            self._play_stop_ticks = 0
            self._seek_guard_until = time.monotonic() + 0.6
            self._apply_play_position(pms)

    def _on_play_state(self, state: str):
        """播放状态变化（轮询驱动，见 _on_play_tick；不再由信号触发）。"""
        if state == "playing":
            self.btn_play.setText("⏸ 暂停")
        else:
            self.btn_play.setText("▶ 播放")
            if state == "stopped":
                if self.slider_pos is not None:
                    self.slider_pos.setValue(0)
                self._play_current_idx = -1
                self._clear_highlight()
                self._stop_highlight_timer()   # 自然播完/停止后停掉轮询

    def _on_play_tick(self):
        """高亮定时器每次触发：状态轮询（替代 playbackStateChanged 信号）+ 位置高亮。

        修复（2026-09-04）：PyQt6 6.11 + Qt6.11 下只要连接了 playbackStateChanged，
        带音频输出的 play() 必然触发 Qt6Core 崩溃（0xc0000409）闪退（连接类型无关，
        连位置/时长/错误信号均无碍）。故改为每 300ms 读一次 playback_state()。
        状态机防抖：加载期/启动瞬间播放器仍报 stopped，只有“接近时长末尾的 stopped”
        或“连续 5 拍(~1.5s) stopped”才算真正停止，避免误触发 UI 复位与高亮清空。
        """
        if self.player is None:
            return
        st = self.player.playback_state()
        last = self._play_state_last
        if st == "playing":
            self._play_stop_ticks = 0
            if last != "playing":
                self._play_state_last = "playing"
                self._on_play_state("playing")
        elif st == "paused":
            self._play_stop_ticks = 0
            if last != "paused":
                self._play_state_last = "paused"
                self._on_play_state("paused")
        else:  # stopped
            d = self.player.duration() if self.player else 0
            pos = self.player.position() if self.player else 0
            if last == "playing":
                near_end = d > 0 and pos >= d - 150
                if near_end:
                    self._play_state_last = "stopped"
                    self._on_play_state("stopped")   # 自然播完
                else:
                    self._play_stop_ticks += 1
                    if self._play_stop_ticks >= 5:   # 卡住/异常停止兜底
                        self._play_state_last = "stopped"
                        self._on_play_state("stopped")
            # last==stopped 且未在播：媒体加载中或尚未真正播放，忽略以免闪 UI
        # 兜底刷新进度条/时间：即使 positionChanged 在某些后端不触发，进度条也能跟随。
        # 刚 seek 过的窗口内（_seek_guard_until）不读 position() 回刷，避免后端异步 seek
        # 尚未到位时用旧值把进度条拉回（造成"点时间戳进度条不跟随/闪回"）。
        if self.player is not None:
            now = time.monotonic()
            if now < self._seek_guard_until:
                # guard 内：沿用 _apply_play_position 的最后写入值（即 seek 目标），不覆盖
                return
            self._seek_guard_until = 0.0
            self._apply_play_position(self.player.position())

    # ---------- 播放高亮 ----------
    def _start_highlight_timer(self):
        if self._play_highlighter_timer is None:
            self._play_highlighter_timer = QTimer(self)
            self._play_highlighter_timer.setInterval(300)
            # tick = 状态轮询（替代已废弃的 playbackStateChanged 信号）+ 位置高亮
            self._play_highlighter_timer.timeout.connect(self._on_play_tick)
        self._play_highlighter_timer.start()

    def _stop_highlight_timer(self):
        if self._play_highlighter_timer is not None:
            self._play_highlighter_timer.stop()

    def _highlight_current_sentence(self, pos_ms: int):
        """根据播放位置高亮当前句（找到则滚动 + extraSelection 黄底）。"""
        meta = self._sentences_meta
        if not meta or self.txt_transcript is None:
            return
        idx = -1
        for i, m in enumerate(meta):
            if m["start_ms"] <= pos_ms < (m["end_ms"] if m["end_ms"] > m["start_ms"] else m["start_ms"] + 1):
                idx = i
                break
        if idx == self._play_current_idx:
            return
        self._play_current_idx = idx
        if idx < 0:
            self._clear_highlight()
            return
        m = self._sentences_meta[idx]
        # 定位该句所在 block 并整行高亮
        cur = self._find_sentence_cursor(m["start_ms"])
        if cur is None:
            return
        # 高亮整个 block（句段）
        fmt = QTextCharFormat()
        fmt.setBackground(QColor(255, 200, 60, 70))
        sel = self.txt_transcript.textCursor()
        sel.setPosition(cur.selectionStart())
        sel.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
        es = QTextEdit.ExtraSelection()
        es.cursor = sel
        es.format = fmt
        self.txt_transcript.setExtraSelections([es])
        # 滚动到可见
        self.txt_transcript.setTextCursor(cur)
        self.txt_transcript.ensureCursorVisible()

    def _find_sentence_cursor(self, start_ms: int):
        """按句子起始时间戳文本定位 QTextCursor（找不到返回 None）。"""
        hh, mm, ss = (start_ms // 3600000, (start_ms % 3600000) // 60000,
                      (start_ms % 60000) // 1000)
        anchor = f"{hh:02d}:{mm:02d}:{ss:02d}"
        doc = self.txt_transcript.document()
        c = doc.find(anchor)
        if not c.isNull():
            return c
        return None

    def _scroll_to_sentence(self, idx: int):
        """滚动视图使第 idx 句可见（兼容旧逻辑，现由高亮统一处理）。"""
        m = self._sentences_meta[idx]
        cur = self._find_sentence_cursor(m["start_ms"])
        if cur is not None:
            self.txt_transcript.setTextCursor(cur)
            self.txt_transcript.ensureCursorVisible()


    def _clear_highlight(self):
        if self.txt_transcript is not None:
            self.txt_transcript.setExtraSelections([])
            cur = self.txt_transcript.textCursor()
            cur.clearSelection()
            self.txt_transcript.setTextCursor(cur)

    def _reset_play_ui(self):
        """切换到新记录/新音频时重置播放条状态。"""
        if self.player is not None:
            self.player.stop()
        self._play_source_path = ""
        self._play_current_idx = -1
        self._play_state_last = "stopped"
        self._play_stop_ticks = 0
        self._pending_seek_ms = None
        self._seek_guard_until = 0.0
        self._stop_highlight_timer()
        if self.slider_pos is not None:
            self.slider_pos.setRange(0, 0)
            self.slider_pos.setValue(0)
            self.slider_pos.setEnabled(False)
        if self.lbl_play_time is not None:
            self.lbl_play_time.setText("00:00 / 00:00")
        if self.btn_play is not None:
            self.btn_play.setText("▶ 播放")

    def _on_hist_new(self):
        """把当前音频手动归档为新记录（未走录音/导入自动建档的场合）。"""
        if self.store is None:
            return
        if not self.current_audio or not Path(self.current_audio).exists():
            NfMessage.info(self, "提示", "当前没有可归档的音频。\n请先录音或导入音频。")
            return
        src = self.current_audio
        try:
            is_wav = Path(src).suffix.lower() == ".wav"
            if not is_wav:
                src = convert_to_wav_16k(src)
            rid = self.store.create(
                title=self.current_audio_name or Path(src).stem,
                source="imported",
            )
            self.store.save_audio(rid, src)
            try:
                import wave as _w
                with _w.open(str(self.store.audio_path(rid)), "rb") as wf:
                    fr = wf.getframerate()
                    n = wf.getnframes()
                    if fr:
                        rec = self.store.get(rid)
                        if rec:
                            rec["duration_sec"] = round(n / fr, 2)
                            self.store._add_to_index(rec)
            except Exception:
                pass
            self._refresh_hist_list()
            self.lbl_status.setText(f"已新建记录：{rid}")
            logger.info(f"手动新建记录: {rid} src={src}")
        except Exception as e:
            logger.error(f"新建记录失败: {e}")
            NfMessage.warn(self, "新建失败", str(e))

    def _auto_archive(self, wav_path: str, source: str = "recorded", title: str = ""):
        """录音/导入完成后自动归档为一条历史记录。

        - 复制 16k wav 进记录目录（播放/转写对齐）
        - 更新索引，刷新左侧列表并选中新记录
        """
        if self.store is None:
            return
        try:
            if not wav_path or not Path(wav_path).exists():
                return
            dur = 0.0
            try:
                import wave as _w
                with _w.open(wav_path, "rb") as wf:
                    fr = wf.getframerate()
                    n = wf.getnframes()
                    if fr:
                        dur = n / fr
            except Exception:
                pass
            rid = self.store.create(
                title=title or Path(wav_path).stem,
                source=source,
                duration_sec=dur,
            )
            self.store.save_audio(rid, wav_path)
            self.current_record_id = rid
            self._refresh_hist_list()
            # 高亮新记录
            item = self._find_hist_item(rid)
            if item:
                self.hist_list.setCurrentItem(item)
            # 阶段 C：新记录有音频 -> 启用播放条
            self._reset_play_ui()
            if self.btn_play is not None:
                self.btn_play.setEnabled(True)
                self.btn_play_stop.setEnabled(True)
            logger.info(f"自动归档记录: {rid} source={source} dur={dur:.1f}s")
        except Exception as e:
            logger.error(f"自动归档失败: {e}")

    # ---------- 事件：录音 ----------
    def on_record_clicked(self):
        logger.info("[按钮] 点击「开始录音」")
        if not PYAUDIO_OK:
            NfMessage.warn(self, "提示", "pyaudio 未安装，无法录音。请导入音频文件代替。")
            return
        if self.recorder is not None and self.recorder.isRunning():
            logger.warning("录音已在进行中")
            return
        self.recorder = AudioRecorder()
        self.recorder.started.connect(self._on_record_started)
        self.recorder.stopped.connect(self._on_record_stopped)
        self.recorder.error.connect(self._on_record_error)
        self.recorder.start()

    def _on_record_started(self):
        logger.info("录音已启动")
        self.record_start_time = time.time()
        self.btn_record.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_duration.setText("时长：00:00:00（录音中…）")
        self.recording_timer = QTimer(self)
        self.recording_timer.timeout.connect(self._update_record_duration)
        self.recording_timer.start(1000)
        self.lbl_status.setText("正在录音…")

    def _update_record_duration(self):
        elapsed = int(time.time() - self.record_start_time)
        self.lbl_duration.setText(f"时长：{format_timestamp(elapsed)}（录音中…）")

    def _on_record_stopped(self, path):
        logger.info(f"录音完成: {path}")
        self.current_audio = path
        self.current_audio_name = Path(path).name
        self.lbl_audio.setText(f"音频文件：{Path(path).name}")
        self.btn_record.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self.recording_timer:
            self.recording_timer.stop()
        self.lbl_status.setText("录音完成")
        # 阶段 B：录音自动归档为一条历史记录
        self._auto_archive(path, source="recorded")

    def _on_record_error(self, msg):
        logger.error(f"录音错误: {msg}")
        self.btn_record.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self.recording_timer:
            self.recording_timer.stop()
        self.lbl_status.setText("录音失败")
        NfMessage.warn(self, "录音失败", f"录音出错：\n{msg}\n\n请检查麦克风设备。")

    def on_stop_clicked(self):
        logger.info("[按钮] 点击「停止录音」")
        if self.recorder is not None and self.recorder.isRunning():
            self.recorder.stop()
            self.lbl_status.setText("正在停止录音…")

    # ---------- 事件：导入 ----------
    def on_import_clicked(self):
        logger.info("[按钮] 点击「导入文件」")
        fd = QFileDialog(self, "选择音频文件", "")
        fd.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        fd.setNameFilter("音频文件 (*.wav *.mp3 *.m4a *.flac);;所有文件 (*.*)")
        if fd.exec() != QFileDialog.DialogCode.Accepted or not fd.selectedFiles():
            return
        fpath = fd.selectedFiles()[0]
        logger.info(f"导入文件: {fpath}")
        try:
            converted = convert_to_wav_16k(fpath)
            self.current_audio = converted
            self.current_audio_name = Path(fpath).name
            self.lbl_audio.setText(f"音频文件：{Path(fpath).name}")
            self._update_audio_duration(converted)
            self.lbl_status.setText(f"已导入: {Path(fpath).name}")
            # 阶段 B：导入自动归档为一条历史记录
            self._auto_archive(converted, source="imported", title=Path(fpath).stem)
        except Exception as e:
            logger.error(f"导入失败: {e}")
            NfMessage.warn(self, "导入失败", str(e))

    def _update_audio_duration(self, path):
        try:
            with wave.open(path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    self.lbl_duration.setText(f"时长：{format_timestamp(frames / rate)}")
                else:
                    self.lbl_duration.setText("时长：--")
        except Exception as e:
            logger.warning(f"读取时长失败: {e}")
            self.lbl_duration.setText("时长：--")

    # ---------- 事件：转写 ----------
    def on_transcribe_clicked(self):
        logger.info("[按钮] 点击「开始转写」")
        if not self.current_audio:
            NfMessage.warn(self, "提示", "请先录音或导入音频文件。")
            return
        if self.transcriber is not None and self.transcriber.isRunning():
            logger.warning("转写已在进行中")
            return
        self._set_busy(True, "已开始转写，请稍候…")
        self.txt_transcript.setPlainText(
            "已开始转写，请稍候…（首次会自动加载语音模型，约需 1 分钟；"
            "加载完成后自动识别，结果会显示在此处）")
        model_dir = self.config.get("model_dir", "mod")
        asr_model = self.config.get("asr_model", "sensevoice")
        vad_level = self.config.get("vad_level", "medium")
        num_spk = int(self.config.get("num_speakers", 0) or 0) or None  # 0=自动
        self.transcriber = TranscriptionWorker(
            audio_path=self.current_audio, model_dir=model_dir,
            audio_name=self.current_audio_name, asr_model=asr_model,
            vad_level=vad_level, num_speakers=num_spk,
        )
        self.transcriber.progress.connect(self._on_transcribe_progress)
        self.transcriber.finished.connect(self._on_transcribe_finished)
        self.transcriber.finished_obj.connect(self._on_transcribe_finished_obj)
        self.transcriber.error.connect(self._on_transcribe_error)
        self.transcriber.start()

    def _on_transcribe_progress(self, text):
        self.lbl_status.setText(text)

    def _on_transcribe_finished(self, transcript):
        logger.info("转写完成回调")
        self.current_transcript = transcript
        header = f"{self.current_audio_name}" if self.current_audio_name else ""
        # 阶段 C：优先用结构化结果渲染句子列表（时间戳可点）
        result_obj = getattr(self, "_pending_result_obj", None)
        if result_obj is not None:
            self._render_sentences(result_obj, header)
        else:
            self._update_sentences_plain(transcript, f"【{header}】" if header else "")
        self._set_busy(False)
        self.lbl_status.setText(f"转写完成：{self.current_audio_name}")
        # 阶段 B：结构化结果存入当前记录
        if result_obj is not None and self.current_record_id and self.store is not None:
            try:
                self.store.save_transcript(self.current_record_id, result_obj)
                self._current_result_obj = result_obj
                self._refresh_hist_list()
                logger.info(f"转写已存入记录: {self.current_record_id}")
            except Exception as e:
                logger.error(f"转写存记录失败: {e}")
        self._pending_result_obj = None
        # 阶段 C：转写后启用播放按钮（若当前记录有音频）
        if self.current_record_id and self.store is not None:
            ap = self.store.audio_path(self.current_record_id)
            if Path(ap).exists():
                self._play_source_path = ""
                if self.btn_play is not None:
                    self.btn_play.setEnabled(True)
                    self.btn_play_stop.setEnabled(True)

    def _on_transcribe_finished_obj(self, result):
        """结构化转写结果（阶段 B：暂存，等 finished 文本回调后统一存）。"""
        self._pending_result_obj = result
        logger.info(f"结构化转写结果收到: {len(result.sentences)} 句")

    def _on_transcribe_error(self, msg):
        logger.error(f"转写错误: {msg}")
        self._pending_result_obj = None
        self._set_busy(False)
        self.txt_transcript.setPlainText(f"[转写失败] {msg}")
        NfMessage.warn(
            self, "转写失败",
            f"转写出错：\n{msg}\n\n请检查模型目录是否正确、显卡驱动是否正常。",
        )

    # ---------- 事件：总结 ----------
    def on_summarize_clicked(self):
        logger.info("[按钮] 点击「生成纪要」")
        transcript = self.current_transcript or self.txt_transcript.toPlainText().strip()
        if not transcript or transcript.startswith("[转写失败]"):
            NfMessage.warn(self, "提示", "请先完成转写。")
            return
        api_key = self.config.get("deepseek_api_key", "")
        if not api_key:
            _old_theme = self.config.get("theme", "light")
            dlg = ConfigDialog(self.config, self)
            if not dlg.exec():
                return
            self._reapply_theme_if_changed(_old_theme)
            api_key = self.config.get("deepseek_api_key", "")
            if not api_key:
                return
        # 总结模板（2026-09-07）：按用户当前选择的模板生成
        _key = self.config.get("summary_template", DEFAULT_SUMMARY_TEMPLATE)
        if _key not in SUMMARY_TEMPLATES:
            _key = DEFAULT_SUMMARY_TEMPLATE
        _prompt = SUMMARY_TEMPLATES[_key][1]
        self._set_busy(True, f"正在生成纪要（{SUMMARY_TEMPLATES[_key][0]}）…")
        self.txt_summary.setPlainText("纪要生成中，请稍候…")
        self.summarizer = SummaryWorker(
            transcript=transcript,
            api_key=api_key,
            base_url=self.config.get("deepseek_base_url", DEFAULT_CONFIG["deepseek_base_url"]),
            model=self.config.get("deepseek_model", DEFAULT_CONFIG["deepseek_model"]),
            prompt_text=_prompt,
        )
        self.summarizer.finished.connect(self._on_summary_finished)
        self.summarizer.error.connect(self._on_summary_error)
        self.summarizer.start()

    def _on_summary_tpl_changed(self, _idx: int):
        """工具栏下拉切换总结模板：立即写入 config，状态栏提示。"""
        _key = self.cmb_summary_tpl.currentData()
        if _key and _key in SUMMARY_TEMPLATES:
            self.config.set("summary_template", _key)
            self.lbl_status.setText(f"已切换总结模板：{SUMMARY_TEMPLATES[_key][0]}")

    def _on_summary_finished(self, summary):
        logger.info("总结完成回调")
        self.txt_summary.setPlainText(summary)
        self._set_busy(False)
        self.lbl_status.setText("纪要生成完成")
        # 阶段 B：纪要存入当前记录
        if self.current_record_id and self.store is not None:
            try:
                self.store.save_summary(self.current_record_id, summary)
                self._refresh_hist_list()
                logger.info(f"纪要已存入记录: {self.current_record_id}")
            except Exception as e:
                logger.error(f"纪要存记录失败: {e}")

    def _on_summary_error(self, msg):
        logger.error(f"总结错误: {msg}")
        self._set_busy(False)
        self.txt_summary.setPlainText(f"[总结失败] {msg}")
        NfMessage.warn(self, "总结失败", f"生成纪要出错：\n{msg}\n\n请检查 API Key 与网络。")

    # ---------- 事件：导出 ----------
    def on_export_clicked(self):
        logger.info("[按钮] 点击「导出纪要」")
        summary = self.txt_summary.toPlainText().strip()
        if not summary or summary.startswith("[总结失败]"):
            NfMessage.warn(self, "提示", "没有可导出的纪要内容。")
            return
        default_name = f"会议纪要_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        fd = QFileDialog(self, "导出会议纪要", default_name)
        fd.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        fd.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        fd.setNameFilter("文本文件 (*.txt)")
        if fd.exec() != QFileDialog.DialogCode.Accepted or not fd.selectedFiles():
            return
        fpath = fd.selectedFiles()[0]
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(summary)
            logger.info(f"纪要已导出: {fpath}")
            self.lbl_status.setText(f"已导出: {fpath}")
            NfMessage.info(self, "导出成功", f"会议纪要已保存到：\n{fpath}")
        except Exception as e:
            logger.error(f"导出失败: {e}")
            NfMessage.warn(self, "导出失败", str(e))

    # ---------- 事件：配置 ----------
    def on_config_clicked(self):
        logger.info("[按钮] 点击「配置」")
        old_model = self.config.get("asr_model", "sensevoice")
        old_dir = self.config.get("model_dir", "mod")
        old_vad = self.config.get("vad_level", "medium")
        old_theme = self.config.get("theme", "light")
        dlg = ConfigDialog(self.config, self)
        dlg.exec()
        new_model = self.config.get("asr_model", old_model)
        new_dir = self.config.get("model_dir", old_dir)
        new_vad = self.config.get("vad_level", old_vad)
        # 主题切换立即生效（含转写区富文本重渲染）
        self._reapply_theme_if_changed(old_theme)
        # 若模型/目录/切句档位变了：重置 manager 并重新预加载，下次转写用新配置
        if new_model != old_model or new_dir != old_dir or new_vad != old_vad:
            logger.info(f"模型配置变更（{old_model}->{new_model}, vad {old_vad}->{new_vad}），"
                        f"重置并重新预加载")
            self._restart_preload_after_config()

    # ---------- 辅助 ----------
    def _set_busy(self, busy, text=""):
        self.progress.setVisible(busy)
        self.btn_transcribe.setEnabled(not busy)
        self.btn_summarize.setEnabled(not busy)
        self.btn_export.setEnabled(not busy)
        self.btn_import.setEnabled(not busy)
        self.btn_record.setEnabled(not busy)
        if hasattr(self, "btn_load_model"):
            # 忙时不禁止加载按钮本体，但防重复：由 _update_load_btn_state 管状态
            self.btn_load_model.setEnabled(not busy)
        if busy:
            self.lbl_status.setText(text)

    # ---------- 无边框窗口：拖动 / 双击最大化 / 按钮（2026-09-07） ----------
    def _titlebar_press(self, e):
        if e.button() == Qt.MouseButton.LeftButton and not self.isMaximized():
            self._drag_off = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
        else:
            self._drag_off = None

    def _titlebar_move(self, e):
        off = self._drag_off
        if off is not None and (e.buttons() & Qt.MouseButton.LeftButton) and not self.isMaximized():
            self.move(e.globalPosition().toPoint() - off)

    def _titlebar_release(self, e):
        self._drag_off = None

    def _titlebar_dblclick(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._toggle_max()

    def _toggle_max(self):
        """最大化/还原切换（自绘标题栏按钮用）。"""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
        self._sync_max_icon()

    def _sync_max_icon(self):
        if getattr(self, "btn_win_max", None) is not None:
            self.btn_win_max.setText("❐" if self.isMaximized() else "□")

    def _apply_theme(self, rerender: bool = True):
        """应用主题（阶段 E：样式全部来自 theme.py）。

        挂在 QApplication 上，使 QDialog / QMenu / QMessageBox 等子窗口同样生效。
        富文本配色同步更新，并按需重渲染转写区（HTML inline 色随主题变化）。
        """
        theme = self.config.get("theme", "light")
        logger.info(f"应用主题: {theme}")
        qss = build_qss(theme)
        self._palette = get_colors(theme)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(qss)
        else:
            self.setStyleSheet(qss)
        # 转写区富文本用的是 inline 颜色，需按新配色重渲染
        if rerender and getattr(self, "_current_result_obj", None) is not None:
            self._render_sentences(self._current_result_obj, self.current_audio_name or "")

    def _reapply_theme_if_changed(self, old_theme: str):
        """配置弹窗关闭后：若主题变了立即整套应用（修复浅色/深色“切了不生效”）。"""
        if self.config.get("theme", "light") != old_theme:
            logger.info(f"主题变更 {old_theme} -> {self.config.get('theme')}，立即应用")
            self._apply_theme()

    def closeEvent(self, event: QCloseEvent):
        logger.info("应用关闭，清理资源…")
        if self.recorder is not None and self.recorder.isRunning():
            self.recorder.stop()
            self.recorder.wait(2000)
        # 阶段 C：释放播放器
        try:
            if self.player is not None:
                self._stop_highlight_timer()
                self.player.release()
        except Exception as e:
            logger.warning(f"释放播放器失败: {e}")
        logger.info("应用已退出")
        event.accept()


# ---------------------------------------------------------------------------
# 11. 程序入口
# ---------------------------------------------------------------------------

def main():
    logger.info("进入 main()")
    if not PYQT_OK:
        logger.critical("PyQt6 不可用")
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0, "缺少 PyQt6 依赖，请先执行：pip install PyQt6", "启动失败", 0x10,
            )
        except Exception:
            pass
        return 1

    app = QApplication(sys.argv)
    app.setApplicationName("智能会议纪要工具")

    click_logger = ClickLogger(app)
    app.installEventFilter(click_logger)

    config = ConfigManager()
    window = MainWindow(config)
    window.show()

    logger.info("主窗口已显示")
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("程序发生未捕获异常")
        raise
