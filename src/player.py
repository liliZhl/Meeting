# -*- coding: utf-8 -*-
"""
player.py — 音频播放封装（阶段 C）

基于 PyQt6.QtMultimedia 的 QMediaPlayer + QAudioOutput。
播放对象是历史记录里的 16k 单声道 wav（与转写时间戳精确对齐）。

对外信号（Qt 信号，跨线程安全）：
  positionChanged(ms:int)
  durationChanged(ms:int)
  errorOccurred(str)
注意：不提供 playbackStateChanged 信号——PyQt6 6.11 + Qt 6.11 下，只要 Python 侧连接了
QMediaPlayer.playbackStateChanged，且播放器已挂音频输出，play() 启动即触发 Qt6Core
内部 fail-fast（0xc0000409）闪退（2026-09-04 实测：连接类型 Auto/Queued 均崩，而连接
positionChanged/durationChanged/errorOccurred 均无恙）。调用方改用 playback_state() 轮询
（配合既有 300ms 高亮定时器即可）。
"""

import logging
from pathlib import Path

from PyQt6.QtCore import QObject, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaDevices, QMediaPlayer

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
    errorOccurred = pyqtSignal(str)        # 播放错误信息（阶段 F 真机诊断）

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ok = QT_MULTIMEDIA_OK
        self._source = ""
        self._player = None
        self._audio_out = None
        self._has_device = False
        if self._ok:
            self._player = QMediaPlayer(self)
            self._audio_out = QAudioOutput(self)
            self._audio_out.setVolume(0.8)
            self._rebind_audio_device()   # 修复：不显式 setDevice 时 play() 会闪退（见下）
            self._player.setAudioOutput(self._audio_out)
            # 设备热插拔/变更（含远程音频接入）时自动重绑默认输出设备
            try:
                QMediaDevices.audioOutputsChanged.connect(self._rebind_audio_device)
            except Exception:
                pass
            # 转发信号（做一次状态字符串翻译）
            # 注意：刻意【不】连接 playbackStateChanged —— PyQt6 6.11 下连接该信号后
            # play() 即触发 Qt6Core 崩溃（0xc0000409），详见模块 docstring。
            self._player.positionChanged.connect(self._on_position)
            self._player.durationChanged.connect(self._on_duration)
            self._player.errorOccurred.connect(self._on_error)

    def _rebind_audio_device(self):
        """把 QAudioOutput 显式绑定到当前默认音频输出设备（设备管理，防御性）。

        说明：真正导致「播放闪退」的根因是连接了 playbackStateChanged 信号（见模块
        docstring），而非设备未绑定；此处显式绑定默认设备 + 监听 audioOutputsChanged
        热插拔重绑，保证设备变更/接入时播放器始终指向有效设备。
        """
        try:
            dev = QMediaDevices.defaultAudioOutput()
            if not dev.isNull() and self._audio_out is not None:
                self._audio_out.setDevice(dev)
                self._has_device = True
            else:
                self._has_device = False
                logger.warning("未检测到可用的音频输出设备（播放将被禁用）")
        except Exception as e:
            self._has_device = False
            logger.warning(f"绑定音频输出设备失败: {e}")

    # ---------- 内部转发 ----------
    def _on_position(self, ms: int):
        self.positionChanged.emit(int(ms))

    def _on_duration(self, ms: int):
        logger.info(f"[播放诊断] durationChanged: {int(ms)} ms, source={self._source}")
        self.durationChanged.emit(int(ms))

    def _on_error(self, err, err_str: str):
        code = getattr(err, "value", err)   # PyQt6 6.11 scoped enum 用 .value，勿 int() 强转
        msg = f"播放错误(code={code}): {err_str or '(无详细信息)'}"
        logger.error(f"[播放诊断] {msg}")
        self.errorOccurred.emit(err_str or f"错误码 {code}")

    # ---------- 对外 API ----------
    @property
    def available(self) -> bool:
        return self._ok

    @property
    def device_ok(self) -> bool:
        """当前是否有可用的音频输出设备（无设备时 play 应被拦截而非崩溃）。"""
        return bool(self._has_device)

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
        if not (self._ok and self._source):
            return
        if not self._has_device:
            logger.warning("无音频输出设备，忽略播放请求")
            self.errorOccurred.emit("未检测到音频输出设备，无法播放")
            return
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

    def playback_state(self) -> str:
        """轮询当前播放状态：stopped | paused | playing。

        替代 playbackStateChanged 信号（PyQt6 6.11 连接该信号会在 play() 时崩溃），
        供调用方在既有定时器里轮询。
        """
        if not self._ok or self._player is None:
            return "stopped"
        try:
            ps = self._player.playbackState()
            # PyQt6 6.11 枚举为 PEP435 scoped enum，不能 int() 强转，用枚举比较
            if ps == QMediaPlayer.PlaybackState.PlayingState:
                return "playing"
            if ps == QMediaPlayer.PlaybackState.PausedState:
                return "paused"
        except Exception:
            pass
        return "stopped"

    def release(self):
        """释放播放器（应用关闭时调用）。"""
        if self._ok:
            try:
                self._player.stop()
                self._player.setSource(QUrl())
            except Exception:
                pass
