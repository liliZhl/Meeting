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
# 2. 依赖导入（优雅降级：缺失时记录警告，界面仍可启动）
# ---------------------------------------------------------------------------

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QTextEdit, QLabel, QFileDialog, QMessageBox, QDialog,
        QLineEdit, QComboBox, QDialogButtonBox, QProgressDialog, QFrame, QStatusBar,
        QProgressBar, QFormLayout, QPlainTextEdit, QSplitter,
        QListWidget, QListWidgetItem, QAbstractItemView, QInputDialog, QMenu,
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
    "asr_model": "fun-asr-nano",  # 识别模型: fun-asr-nano / sensevoice / paraformer
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

    def __init__(self, audio_path: str, model_dir: str, audio_name: str = "",
                 asr_model: str = None, vad_level: str = None):
        super().__init__()
        self.audio_path = audio_path
        self.model_dir = model_dir
        self.audio_name = audio_name or Path(audio_path).name
        self.asr_model = asr_model
        self.vad_level = vad_level

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
        self.setMinimumWidth(560)
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

        # 语音识别模型选择（延迟 import，避免启动卡顿）
        from asr_engine import ASR_MODELS, MODEL_NANO
        self.asr_model_combo = QComboBox()
        self._asr_model_keys = list(ASR_MODELS.keys())
        for k in self._asr_model_keys:
            self.asr_model_combo.addItem(ASR_MODELS[k]["label"], k)
        self.asr_model_combo.setToolTip(
            "识别引擎：Fun-ASR-Nano 需 GPU/强 CPU（质量最高）；\n"
            "SenseVoice / Paraformer 体积小、CPU 友好，适合无独显电脑。\n"
            "切换后需重启应用生效（首次加载新模型需下载对应文件）。"
        )
        form.addRow("语音识别模型：", self.asr_model_combo)

        # 模型说明（随选择更新）
        self.asr_model_desc = QLabel()
        self.asr_model_desc.setStyleSheet("color: gray; font-size: 11px;")
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

    def _on_asr_model_changed(self, idx: int):
        from asr_engine import ASR_MODELS
        if 0 <= idx < len(self._asr_model_keys):
            self.asr_model_desc.setText(ASR_MODELS[self._asr_model_keys[idx]]["desc"])

    def _load_values(self):
        self.api_key_edit.setText(self.config.get("deepseek_api_key", ""))
        self.base_url_edit.setText(self.config.get("deepseek_base_url", DEFAULT_CONFIG["deepseek_base_url"]))
        self.model_edit.setText(self.config.get("deepseek_model", DEFAULT_CONFIG["deepseek_model"]))
        self.model_dir_edit.setText(self.config.get("model_dir", DEFAULT_CONFIG["model_dir"]))
        # 恢复已保存的识别模型选择
        cur = self.config.get("asr_model", "")
        idx = self._asr_model_keys.index(cur) if cur in self._asr_model_keys else 0
        self.asr_model_combo.setCurrentIndex(idx)
        self._on_asr_model_changed(idx)
        # 恢复已保存的切句灵敏度
        cur_vad = self.config.get("vad_level", "")
        idx_vad = self._vad_keys.index(cur_vad) if cur_vad in self._vad_keys else 0
        self.vad_combo.setCurrentIndex(idx_vad)

    def _on_accept(self):
        key = self.api_key_edit.text().strip()
        if not key:
            QMessageBox.warning(self, "提示", "API Key 不能为空。")
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
        self.config.save()
        logger.info(f"配置已更新（asr_model={cur}, vad_level={cur_vad}）")
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

        # 历史记录存储（阶段 B）
        try:
            from store import RecordStore
            self.store = RecordStore(str(RECORDS_DIR))
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
            QMessageBox.warning(
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
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)
        hist_title = QLabel("🗂 历史记录")
        hist_title.setObjectName("histTitle")
        hist_title.setStyleSheet("font-weight: bold; padding: 2px;")
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
    def _check_model_status(self) -> str:
        """返回模型状态文本：就绪的模型名 / 未就绪原因。

        按配置中的 asr_model 检查对应主模型是否存在（VAD/说话人公共组件
        由引擎加载时统一校验）。返回的字符串供状态栏显示。
        """
        try:
            from asr_engine import ASR_MODELS
        except Exception:
            return "⚠️ 模型模块不可用"
        model_key = self.config.get("asr_model", "fun-asr-nano")
        sub = model_key if model_key in ASR_MODELS else "fun-asr-nano"
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
        QMessageBox.information(
            self, "欢迎使用",
            "欢迎使用智能会议纪要工具！\n\n"
            "首次使用请先配置 DeepSeek API Key（用于生成会议纪要）。\n"
            "语音识别模型放在程序目录的 mod/ 文件夹内。",
        )
        dlg = ConfigDialog(self.config, self)
        dlg.exec()
        # 配置弹窗可能修改了识别模型：重置并重新预加载
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
                self.txt_transcript.setPlainText(f"【{self.current_audio_name}】\n" + self.current_transcript)
                self._current_result_obj = tr
            except Exception as e:
                logger.warning(f"转写解析失败: {e}")
                self.txt_transcript.setPlainText("(该记录暂无可用转写内容)")
                self.current_transcript = ""
                self._current_result_obj = None
        else:
            self.current_transcript = ""
            self._current_result_obj = None
            self.txt_transcript.setPlainText("该记录尚未转写。\n点击「🔊 开始转写」进行语音转写。")
        summary = self.store.load_summary(rid)
        if summary:
            self.txt_summary.setPlainText(summary)
        else:
            self.txt_summary.setPlainText("该记录尚未生成纪要。\n点击「✨ 生成纪要」调用 AI 总结。")
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
            new_title, ok = QInputDialog.getText(
                self, "重命名记录", "新名称：", text=rec.get("title", "")
            )
            if ok and new_title.strip():
                self.store.rename(rid, new_title.strip())
                self._refresh_hist_list()
                if self.current_record_id == rid:
                    self.current_audio_name = new_title.strip()
                    self.lbl_status.setText(f"已重命名为：{new_title.strip()}")
                logger.info(f"重命名记录: {rid} -> {new_title.strip()}")
        elif act == act_delete:
            ret = QMessageBox.question(
                self, "删除记录",
                f"确定删除记录「{rec.get('title', rid)}」？\n将删除其音频、转写与纪要文件。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ret == QMessageBox.StandardButton.Yes:
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
        self.lbl_audio.setText("音频文件：无")
        self.lbl_duration.setText("时长：--")
        self.txt_transcript.clear()
        self.txt_summary.clear()

    def _on_hist_new(self):
        """把当前音频手动归档为新记录（未走录音/导入自动建档的场合）。"""
        if self.store is None:
            return
        if not self.current_audio or not Path(self.current_audio).exists():
            QMessageBox.information(self, "提示", "当前没有可归档的音频。\n请先录音或导入音频。")
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
            QMessageBox.warning(self, "新建失败", str(e))

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
            logger.info(f"自动归档记录: {rid} source={source} dur={dur:.1f}s")
        except Exception as e:
            logger.error(f"自动归档失败: {e}")

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
        # 阶段 B：录音自动归档为一条历史记录
        self._auto_archive(path, source="recorded")

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
            # 阶段 B：导入自动归档为一条历史记录
            self._auto_archive(converted, source="imported", title=Path(fpath).stem)
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
        asr_model = self.config.get("asr_model", "fun-asr-nano")
        vad_level = self.config.get("vad_level", "medium")
        self.transcriber = TranscriptionWorker(
            audio_path=self.current_audio, model_dir=model_dir,
            audio_name=self.current_audio_name, asr_model=asr_model,
            vad_level=vad_level,
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
        header = f"【{self.current_audio_name}】\n" if self.current_audio_name else ""
        self.txt_transcript.setPlainText(header + transcript)
        self._set_busy(False)
        self.lbl_status.setText(f"转写完成：{self.current_audio_name}")
        # 阶段 B：结构化结果存入当前记录
        result_obj = getattr(self, "_pending_result_obj", None)
        if result_obj is not None and self.current_record_id and self.store is not None:
            try:
                self.store.save_transcript(self.current_record_id, result_obj)
                self._current_result_obj = result_obj
                self._refresh_hist_list()
                logger.info(f"转写已存入记录: {self.current_record_id}")
            except Exception as e:
                logger.error(f"转写存记录失败: {e}")
        self._pending_result_obj = None

    def _on_transcribe_finished_obj(self, result):
        """结构化转写结果（阶段 B：暂存，等 finished 文本回调后统一存）。"""
        self._pending_result_obj = result
        logger.info(f"结构化转写结果收到: {len(result.sentences)} 句")

    def _on_transcribe_error(self, msg):
        logger.error(f"转写错误: {msg}")
        self._pending_result_obj = None
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
        old_model = self.config.get("asr_model", "fun-asr-nano")
        old_dir = self.config.get("model_dir", "mod")
        old_vad = self.config.get("vad_level", "medium")
        dlg = ConfigDialog(self.config, self)
        dlg.exec()
        new_model = self.config.get("asr_model", old_model)
        new_dir = self.config.get("model_dir", old_dir)
        new_vad = self.config.get("vad_level", old_vad)
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
