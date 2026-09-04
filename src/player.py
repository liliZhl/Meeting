# -*- coding: utf-8 -*-
"""
player.py — 音频播放封装（阶段 C）

基于 PyQt6.QtMultimedia 的 QMediaPlayer + QAudioOutput。
播放对象是历史记录里的 16k 单声道 wav（与转写时间戳精确对齐）。

对外信号（Qt 信号，跨线程安全）：
  positionChanged(ms:int)
  durationChanged(ms:int)
  stateChanged(str)      # stopped | playing | paused
"""

import logging
from pathlib import Path

from PyQt6.QtCore import QObject, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer

logger = logging.getLogger("MeetingAssistant.Player")

# QtMultimedia 是否可用（导入失败时播放功能降级隐藏）
QT_MULTIMEDIA_OK = True
try:
    from PyQt6.QtMultimedia import QMediaPlayer as _QMP  # noqa: F401
except Exception:  # pragma: no cover
    QT_MULTIMEDIA_OK = False
    logger.warning("QtMultimedia 不可用，播放功能将禁用")


class AudioPlayer(QObject):
    """封装 QMediaPlayer 播放本地音频。"""

    positionChanged = pyqtSignal(int)      # 播放位置 ms
    durationChanged = pyqtSignal(int)      # 总时长 ms
    stateChanged = pyqtSignal(str)         # stopped | playing | paused
    errorOccurred = pyqtSignal(str)        # 播放错误信息（阶段 F 真机诊断）

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ok = QT_MULTIMEDIA_OK
        self._source = ""
        self._player = None
        self._audio_out = None
        if self._ok:
            self._player = QMediaPlayer(self)
            self._audio_out = QAudioOutput(self)
            self._audio_out.setVolume(0.8)
            self._player.setAudioOutput(self._audio_out)
            # 转发信号（做一次状态字符串翻译）
            self._player.positionChanged.connect(self._on_position)
            self._player.durationChanged.connect(self._on_duration)
            self._player.playbackStateChanged.connect(self._on_state)
            self._player.errorOccurred.connect(self._on_error)

    # ---------- 内部转发 ----------
    def _on_position(self, ms: int):
        self.positionChanged.emit(int(ms))

    def _on_duration(self, ms: int):
        logger.info(f"[播放诊断] durationChanged: {int(ms)} ms, source={self._source}")
        self.durationChanged.emit(int(ms))

    def _on_state(self, state):
        name = {0: "stopped", 1: "paused", 2: "playing"}.get(int(state), "stopped")
        self.stateChanged.emit(name)

    def _on_error(self, err, err_str: str):
        msg = f"播放错误(code={int(err)}): {err_str or '(无详细信息)'}"
        logger.error(f"[播放诊断] {msg}")
        self.errorOccurred.emit(err_str or f"错误码 {int(err)}")

    # ---------- 对外 API ----------
    @property
    def available(self) -> bool:
        return self._ok

    @property
    def source(self) -> str:
        return self._source

    def set_source(self, path) -> bool:
        """设置播放源（本地 wav/mp3 等）。成功返回 True。"""
        if not self._ok:
            return False
        p = Path(path)
        if not p.exists():
            logger.warning(f"播放源不存在: {p}")
            return False
        self._source = str(p)
        self._player.setSource(QUrl.fromLocalFile(str(p)))
        return True

    def play(self):
        if self._ok and self._source:
            self._player.play()

    def pause(self):
        if self._ok:
            self._player.pause()

    def stop(self):
        if self._ok:
            self._player.stop()

    def seek(self, ms: int):
        """跳转到指定位置（毫秒）。"""
        if self._ok and self._source:
            self._player.setPosition(max(0, int(ms)))

    def position(self) -> int:
        return int(self._player.position()) if self._ok else 0

    def duration(self) -> int:
        return int(self._player.duration()) if self._ok else 0

    def is_playing(self) -> bool:
        return self._ok and self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    def release(self):
        """释放播放器（应用关闭时调用）。"""
        if self._ok:
            try:
                self._player.stop()
                self._player.setSource(QUrl())
            except Exception:
                pass
