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
# 2. 依赖导入（优雅降级：缺失时记录警告，界面仍可启动）
# ---------------------------------------------------------------------------

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QTextEdit, QLabel, QFileDialog, QMessageBox, QDialog,
        QLineEdit, QDialogButtonBox, QProgressDialog, QFrame, QStatusBar,
        QProgressBar, QFormLayout, QPlainTextEdit, QSplitter,
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject, QEvent
    from PyQt6.QtGui import QFont, QTextCursor, QCloseEvent
    PYQT_OK = True
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

FFMPEG_AVAILABLE = FFMPEG_PATH.exists()


# ---------------------------------------------------------------------------
# 3. 配置管理
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "deepseek_api_key": "",
    "deepseek_base_url": "https://api.deepseek.com",
    "deepseek_model": "deepseek-chat",
    "theme": "dark",
    "model_dir": "mod",           # 模型目录（相对 EXE 或绝对路径）
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
            if FFMPEG_PATH.exists():
                AudioSegment.converter = str(FFMPEG_PATH)
                AudioSegment.ffmpeg = str(FFMPEG_PATH)
                AudioSegment.ffprobe = str(FFMPEG_PATH)
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
            cmd = [str(FFMPEG_PATH), "-y", "-i", str(src),
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

    def __init__(self, audio_path: str, model_dir: str, audio_name: str = ""):
        super().__init__()
        self.audio_path = audio_path
        self.model_dir = model_dir
        self.audio_name = audio_name or Path(audio_path).name

    def run(self):
        try:
            from model_manager import get_manager
            # 单例 manager：若后台尚未加载完成则同步等待
            manager = get_manager(model_root=self.model_dir)
            if not manager.is_ready():
                self.progress.emit("正在加载语音模型（首次约需 1 分钟，请耐心等待）…")
            else:
                self.progress.emit("模型已就绪，开始转写…")
            self.progress.emit(f"正在转写「{self.audio_name}」（音频较长时可能需要几分钟）…")
            result = manager.transcribe(
                self.audio_path, progress_callback=self._on_progress,
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

SUMMARY_PROMPT = """你是一名专业的会议纪要助理。请根据以下转录文本（含说话人和时间戳）生成结构化的会议纪要。

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
{transcript}
"""


class SummaryWorker(QThread):
    """AI 智能总结。"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, transcript, api_key, base_url, model):
        super().__init__()
        self.transcript = transcript
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def run(self):
        try:
            if not OPENAI_OK:
                raise RuntimeError("openai 库未安装")
            if not self.api_key:
                raise RuntimeError("DeepSeek API Key 为空")
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            prompt = SUMMARY_PROMPT.format(transcript=self.transcript)
            logger.info(f"调用 DeepSeek 总结 (model={self.model})")
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一名专业的会议纪要助理。"},
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
# 9. 配置对话框
# ---------------------------------------------------------------------------

class ConfigDialog(QDialog):
    """配置输入对话框。"""

    def __init__(self, config: ConfigManager, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("配置 - 智能会议纪要工具")
        self.setMinimumWidth(520)
        self._build_ui()
        self._load_values()

    def _build_ui(self):
        layout = QVBoxLayout(self)
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

        layout.addLayout(form)

        tip = QLabel("提示：API Key 仅保存在本地 config.json，不会上传。")
        tip.setStyleSheet("color: gray;")
        layout.addWidget(tip)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_values(self):
        self.api_key_edit.setText(self.config.get("deepseek_api_key", ""))
        self.base_url_edit.setText(self.config.get("deepseek_base_url", DEFAULT_CONFIG["deepseek_base_url"]))
        self.model_edit.setText(self.config.get("deepseek_model", DEFAULT_CONFIG["deepseek_model"]))
        self.model_dir_edit.setText(self.config.get("model_dir", DEFAULT_CONFIG["model_dir"]))

    def _on_accept(self):
        key = self.api_key_edit.text().strip()
        if not key:
            QMessageBox.warning(self, "提示", "API Key 不能为空。")
            return
        self.config.set("deepseek_api_key", key)
        self.config.set("deepseek_base_url", self.base_url_edit.text().strip())
        self.config.set("deepseek_model", self.model_edit.text().strip())
        self.config.set("model_dir", self.model_dir_edit.text().strip() or "mod")
        self.config.save()
        logger.info("配置已更新")
        self.accept()


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
        self.recorder = None
        self.transcriber = None
        self.summarizer = None
        self.recording_timer = None
        self.record_start_time = 0.0
        self._model_manager = None
        self._preload_sig = None
        self._preload_poll_timer = None

        self._build_ui()
        self._apply_theme()
        self._check_first_run()
        self._start_model_preload()

    # ---------- 启动预加载模型 ----------
    def _start_model_preload(self):
        """启动后后台预加载 ASR 模型，转写时零等待。

        仅当模型目录就绪时预加载；否则保持 idle，等转写时再处理。
        进度通过定时轮询状态更新到状态栏（后台线程不能直接碰 UI）。
        """
        model_dir = self.config.get("model_dir", "mod")
        mp = Path(model_dir)
        if not mp.is_absolute():
            mp = APP_DIR / model_dir
        if not (mp / "fun-asr-nano" / "model.pt").exists():
            logger.warning("模型目录未就绪，跳过启动预加载")
            return

        try:
            from model_manager import get_manager
            manager = get_manager(model_root=str(mp))
        except Exception as e:
            logger.warning(f"ModelManager 初始化失败，跳过预加载: {e}")
            return

        self._model_manager = manager
        self.lbl_status.setText("正在后台加载模型（可先进行其他操作）…")

        # 进度回调：后台线程 -> 信号 -> 主线程
        if self._preload_sig is None:
            self._preload_sig = _PreloadSignal()
            self._preload_sig.progress.connect(self._on_preload_progress)
            self._preload_sig.state_changed.connect(self._on_preload_state)

        manager.load_async(progress_callback=self._preload_sig.emit_progress)

        # 主线程 QTimer 轮询状态，后台加载完成/失败时刷新状态栏
        from PyQt6.QtCore import QTimer
        self._preload_poll_count = 0
        def _poll():
            if manager is None:
                return
            st = manager.state
            if st in ("ready", "failed"):
                self._on_preload_state(st, manager.error)
                self._preload_poll_timer.stop()
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
        self.setWindowTitle("智能会议纪要工具")
        self.resize(1100, 760)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # 标题
        title = QLabel("智能会议纪要工具")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; padding: 4px;")
        root.addWidget(title)

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
        th.addStretch(1)
        self.btn_transcribe = QPushButton("🔊 开始转写")
        self.btn_transcribe.setObjectName("btnTranscribe")
        th.addWidget(self.btn_transcribe)
        tl.addLayout(th)
        self.txt_transcript = QPlainTextEdit()
        self.txt_transcript.setObjectName("txtTranscript")
        self.txt_transcript.setReadOnly(True)
        self.txt_transcript.setPlaceholderText("转写结果将显示在这里（带时间戳和说话人标签）…")
        tl.addWidget(self.txt_transcript)

        # 纪要区
        sg = QWidget()
        sl = QVBoxLayout(sg)
        sl.setContentsMargins(0, 0, 0, 0)
        sh = QHBoxLayout()
        sh.addWidget(QLabel("📋 会议纪要"))
        sh.addStretch(1)
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
        root.addWidget(splitter, stretch=1)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setObjectName("progressBar")
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        # 状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setObjectName("lblStatus")
        self.statusBar.addWidget(self.lbl_status)

        # 信号连接
        self.btn_record.clicked.connect(self.on_record_clicked)
        self.btn_stop.clicked.connect(self.on_stop_clicked)
        self.btn_import.clicked.connect(self.on_import_clicked)
        self.btn_transcribe.clicked.connect(self.on_transcribe_clicked)
        self.btn_summarize.clicked.connect(self.on_summarize_clicked)
        self.btn_export.clicked.connect(self.on_export_clicked)
        self.btn_config.clicked.connect(self.on_config_clicked)

    # ---------- 首次运行 ----------
    def _check_first_run(self):
        if not self.config.get("deepseek_api_key", ""):
            logger.info("首次运行：API Key 为空，弹出配置引导")
            QTimer.singleShot(300, self._show_first_run_dialog)
        # 检查模型目录
        model_dir = self.config.get("model_dir", "mod")
        model_path = Path(model_dir)
        if not model_path.is_absolute():
            model_path = APP_DIR / model_dir
        if not model_path.exists() or not (model_path / "fun-asr-nano" / "model.pt").exists():
            logger.warning(f"模型目录未就绪: {model_path}")
            self.lbl_status.setText("⚠️ 模型未就绪，请检查 mod 目录")
        else:
            self.lbl_status.setText("✅ 就绪（模型已检测到）")

    def _show_first_run_dialog(self):
        QMessageBox.information(
            self, "欢迎使用",
            "欢迎使用智能会议纪要工具！\n\n"
            "首次使用请先配置 DeepSeek API Key（用于生成会议纪要）。\n"
            "语音识别模型放在程序目录的 mod/ 文件夹内。",
        )
        dlg = ConfigDialog(self.config, self)
        dlg.exec()

    # ---------- 事件：录音 ----------
    def on_record_clicked(self):
        logger.info("[按钮] 点击「开始录音」")
        if not PYAUDIO_OK:
            QMessageBox.warning(self, "提示", "pyaudio 未安装，无法录音。请导入音频文件代替。")
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

    def _on_record_error(self, msg):
        logger.error(f"录音错误: {msg}")
        self.btn_record.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self.recording_timer:
            self.recording_timer.stop()
        self.lbl_status.setText("录音失败")
        QMessageBox.warning(self, "录音失败", f"录音出错：\n{msg}\n\n请检查麦克风设备。")

    def on_stop_clicked(self):
        logger.info("[按钮] 点击「停止录音」")
        if self.recorder is not None and self.recorder.isRunning():
            self.recorder.stop()
            self.lbl_status.setText("正在停止录音…")

    # ---------- 事件：导入 ----------
    def on_import_clicked(self):
        logger.info("[按钮] 点击「导入文件」")
        fpath, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", "",
            "音频文件 (*.wav *.mp3 *.m4a *.flac);;所有文件 (*.*)",
        )
        if not fpath:
            return
        logger.info(f"导入文件: {fpath}")
        try:
            converted = convert_to_wav_16k(fpath)
            self.current_audio = converted
            self.current_audio_name = Path(fpath).name
            self.lbl_audio.setText(f"音频文件：{Path(fpath).name}")
            self._update_audio_duration(converted)
            self.lbl_status.setText(f"已导入: {Path(fpath).name}")
        except Exception as e:
            logger.error(f"导入失败: {e}")
            QMessageBox.warning(self, "导入失败", str(e))

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
            QMessageBox.warning(self, "提示", "请先录音或导入音频文件。")
            return
        if self.transcriber is not None and self.transcriber.isRunning():
            logger.warning("转写已在进行中")
            return
        self._set_busy(True, "正在转写…")
        self.txt_transcript.setPlainText("转写中，请稍候（首次会加载模型，可能需要一点时间）…")
        model_dir = self.config.get("model_dir", "mod")
        self.transcriber = TranscriptionWorker(
            audio_path=self.current_audio, model_dir=model_dir,
            audio_name=self.current_audio_name,
        )
        self.transcriber.progress.connect(self._on_transcribe_progress)
        self.transcriber.finished.connect(self._on_transcribe_finished)
        self.transcriber.error.connect(self._on_transcribe_error)
        self.transcriber.start()

    def _on_transcribe_progress(self, text):
        self.lbl_status.setText(text)

    def _on_transcribe_finished(self, transcript):
        logger.info("转写完成回调")
        self.current_transcript = transcript
        header = f"【{self.current_audio_name}】\n" if self.current_audio_name else ""
        self.txt_transcript.setPlainText(header + transcript)
        self._set_busy(False)
        self.lbl_status.setText(f"转写完成：{self.current_audio_name}")

    def _on_transcribe_error(self, msg):
        logger.error(f"转写错误: {msg}")
        self._set_busy(False)
        self.txt_transcript.setPlainText(f"[转写失败] {msg}")
        QMessageBox.warning(
            self, "转写失败",
            f"转写出错：\n{msg}\n\n请检查模型目录是否正确、显卡驱动是否正常。",
        )

    # ---------- 事件：总结 ----------
    def on_summarize_clicked(self):
        logger.info("[按钮] 点击「生成纪要」")
        transcript = self.current_transcript or self.txt_transcript.toPlainText().strip()
        if not transcript or transcript.startswith("[转写失败]"):
            QMessageBox.warning(self, "提示", "请先完成转写。")
            return
        api_key = self.config.get("deepseek_api_key", "")
        if not api_key:
            dlg = ConfigDialog(self.config, self)
            if not dlg.exec():
                return
            api_key = self.config.get("deepseek_api_key", "")
            if not api_key:
                return
        self._set_busy(True, "正在生成纪要…")
        self.txt_summary.setPlainText("纪要生成中，请稍候…")
        self.summarizer = SummaryWorker(
            transcript=transcript,
            api_key=api_key,
            base_url=self.config.get("deepseek_base_url", DEFAULT_CONFIG["deepseek_base_url"]),
            model=self.config.get("deepseek_model", DEFAULT_CONFIG["deepseek_model"]),
        )
        self.summarizer.finished.connect(self._on_summary_finished)
        self.summarizer.error.connect(self._on_summary_error)
        self.summarizer.start()

    def _on_summary_finished(self, summary):
        logger.info("总结完成回调")
        self.txt_summary.setPlainText(summary)
        self._set_busy(False)
        self.lbl_status.setText("纪要生成完成")

    def _on_summary_error(self, msg):
        logger.error(f"总结错误: {msg}")
        self._set_busy(False)
        self.txt_summary.setPlainText(f"[总结失败] {msg}")
        QMessageBox.warning(self, "总结失败", f"生成纪要出错：\n{msg}\n\n请检查 API Key 与网络。")

    # ---------- 事件：导出 ----------
    def on_export_clicked(self):
        logger.info("[按钮] 点击「导出纪要」")
        summary = self.txt_summary.toPlainText().strip()
        if not summary or summary.startswith("[总结失败]"):
            QMessageBox.warning(self, "提示", "没有可导出的纪要内容。")
            return
        default_name = f"会议纪要_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        fpath, _ = QFileDialog.getSaveFileName(
            self, "导出会议纪要", default_name, "文本文件 (*.txt)",
        )
        if not fpath:
            return
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(summary)
            logger.info(f"纪要已导出: {fpath}")
            self.lbl_status.setText(f"已导出: {fpath}")
            QMessageBox.information(self, "导出成功", f"会议纪要已保存到：\n{fpath}")
        except Exception as e:
            logger.error(f"导出失败: {e}")
            QMessageBox.warning(self, "导出失败", str(e))

    # ---------- 事件：配置 ----------
    def on_config_clicked(self):
        logger.info("[按钮] 点击「配置」")
        dlg = ConfigDialog(self.config, self)
        dlg.exec()

    # ---------- 辅助 ----------
    def _set_busy(self, busy, text=""):
        self.progress.setVisible(busy)
        self.btn_transcribe.setEnabled(not busy)
        self.btn_summarize.setEnabled(not busy)
        self.btn_export.setEnabled(not busy)
        self.btn_import.setEnabled(not busy)
        self.btn_record.setEnabled(not busy)
        if busy:
            self.lbl_status.setText(text)

    def _apply_theme(self):
        theme = self.config.get("theme", "dark")
        logger.info(f"应用主题: {theme}")
        if theme == "dark":
            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #1e1e1e; color: #e0e0e0; }
                QPushButton { background-color: #2d2d2d; color: #e0e0e0;
                    border: 1px solid #444; border-radius: 5px; padding: 6px 14px; }
                QPushButton:hover { background-color: #3a3a3a; }
                QPushButton:disabled { background-color: #2a2a2a; color: #666; }
                QPlainTextEdit { background-color: #252526; color: #d4d4d4;
                    border: 1px solid #3c3c3c; border-radius: 4px;
                    font-family: 'Consolas','Microsoft YaHei'; font-size: 12px; }
                QLabel { color: #e0e0e0; }
                QLineEdit { background-color: #2d2d2d; color: #e0e0e0;
                    border: 1px solid #444; border-radius: 4px; padding: 4px; }
                QStatusBar { background-color: #007acc; color: white; }
                QProgressBar { border: 1px solid #444; border-radius: 4px; text-align: center; }
                QProgressBar::chunk { background-color: #007acc; }
            """)
        else:
            self.setStyleSheet("")

    def closeEvent(self, event: QCloseEvent):
        logger.info("应用关闭，清理资源…")
        if self.recorder is not None and self.recorder.isRunning():
            self.recorder.stop()
            self.recorder.wait(2000)
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
