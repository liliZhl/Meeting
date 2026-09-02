# -*- coding: utf-8 -*-
"""
ModelManager — ASR 模型常驻管理与后台预加载（单例）

背景：
  旧版每次转写在 TranscriptionWorker 里新建 ASREngine 并 load_model，
  线程结束后 engine 被回收，导致每次转写都重新加载模型（约 60 秒）。

本模块：
  - 全局唯一 ASREngine 实例，常驻内存（首次加载后复用）
  - 应用启动后后台线程预加载，转写时零等待
  - 线程安全：加载/推理通过同一把锁串行化
  - 状态机：idle -> loading -> ready | failed

设计约束：
  - 不依赖 Qt（纯 Python），便于单测与复用
  - 通过回调把进度/状态变化抛给 UI（ModelManager 不持有 UI 引用）
"""

import logging
import threading

logger = logging.getLogger("MeetingAssistant.ModelManager")


class ModelManager:
    """ASR 模型单例管理器（线程安全）。"""

    # 状态常量
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"

    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls, model_root: str = None, device: str = None,
                 asr_model: str = None) -> "ModelManager":
        """获取全局单例。首次调用可指定 model_root/device/asr_model。"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(model_root=model_root, device=device,
                                        asr_model=asr_model)
        return cls._instance

    def __init__(self, model_root: str = None, device: str = None,
                 asr_model: str = None):
        self._lock = threading.Lock()          # 串行化 加载/推理
        self._state_lock = threading.Lock()    # 保护状态字段
        self._engine = None
        self._state = self.IDLE
        self._error = ""
        self._load_thread = None
        # 延迟 import ASREngine（模块加载轻量，避免循环依赖）
        from asr_engine import ASREngine
        self._engine_cls = ASREngine
        # 预解析默认参数
        self._model_root = model_root
        self._device = device
        self._asr_model = asr_model
        # 就绪事件：等待模型就绪的线程在此阻塞
        self._ready_event = threading.Event()

    # ---- 状态访问 ----
    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    @property
    def error(self) -> str:
        with self._state_lock:
            return self._error

    @property
    def asr_model(self) -> str:
        """当前引擎所用的主识别模型名（未加载时返回配置值）。"""
        if self._engine is not None:
            return getattr(self._engine, "asr_model", self._asr_model)
        return self._asr_model

    @property
    def engine(self):
        return self._engine

    def is_ready(self) -> bool:
        return self.state == self.READY and self._engine is not None

    # ---- 状态更新 ----
    def _set_state(self, state: str, error: str = ""):
        with self._state_lock:
            old = self._state
            self._state = state
            self._error = error
        logger.info(f"ModelManager 状态: {old} -> {state}"
                    + (f"（错误: {error}）" if error else ""))

    # ---- 模型加载 ----
    def load_async(self, progress_callback=None):
        """
        后台线程预加载模型（幂等：已就绪/加载中则直接返回）。

        progress_callback: 可选，接收 (step:str, detail:str)，
                           会在后台线程中回调，UI 需自行切主线程。
        """
        if self.is_ready():
            logger.info("模型已就绪，跳过预加载")
            return
        if self.state == self.LOADING:
            logger.info("模型加载中，跳过")
            return
        # 需要真正开始加载
        if self._load_thread is not None and self._load_thread.is_alive():
            logger.info("已有加载线程在跑")
            return

        self._load_thread = threading.Thread(
            target=self._load_worker,
            args=(progress_callback,),
            daemon=True,
            name="ModelManager-loader",
        )
        self._load_thread.start()

    def load_sync(self, progress_callback=None) -> bool:
        """同步加载（阻塞调用线程），返回是否成功。"""
        if self.is_ready():
            return True
        if self._load_thread is not None and self._load_thread.is_alive():
            # 已有后台线程在加载，等待它完成
            self._ready_event.wait(timeout=600)
            return self.is_ready()
        return self._load_worker(progress_callback)

    def _load_worker(self, progress_callback=None) -> bool:
        """真正的加载逻辑（在持锁线程中执行）。"""
        with self._lock:
            if self.is_ready():
                return True
            if self.state == self.LOADING:
                # 另一处已在加载：等就绪事件
                self._ready_event.wait(timeout=600)
                return self.is_ready()
            self._ready_event.clear()
            self._set_state(self.LOADING)
            try:
                engine = self._engine_cls(
                    model_root=self._model_root,
                    device=self._device,
                    asr_model=self._asr_model,
                )
                engine.load_model(progress_callback=progress_callback)
                self._engine = engine
                self._set_state(self.READY)
                return True
            except Exception as e:
                logger.exception("模型后台加载失败")
                self._set_state(self.FAILED, str(e))
                return False
            finally:
                self._ready_event.set()

    # ---- 转写（复用常驻引擎） ----
    def transcribe(self, audio_path: str, progress_callback=None):
        """
        使用常驻引擎转写（结构化结果）。

        若模型未就绪：阻塞等待后台加载（最多 ~600s），失败抛 RuntimeError。
        线程安全：全程持锁，避免并发转写冲突。
        """
        if not self.is_ready():
            ok = self.load_sync(progress_callback=progress_callback)
            if not ok:
                raise RuntimeError(f"模型未就绪：{self.error or '加载失败'}")
        with self._lock:
            if self._engine is None:
                raise RuntimeError("模型引擎不可用")
            return self._engine.transcribe(
                audio_path, progress_callback=progress_callback,
            )

    # ---- 重置（测试/换模型目录用） ----
    def reset(self):
        """清空引擎与状态（下次调用重新加载）。"""
        with self._lock:
            self._engine = None
            self._set_state(self.IDLE)
            self._ready_event.clear()


# 模块级便捷引用
_default_manager = None
_default_lock = threading.Lock()


def get_manager(model_root: str = None, device: str = None,
                asr_model: str = None) -> ModelManager:
    """获取全局默认 ModelManager（供主程序使用）。"""
    global _default_manager
    if _default_manager is None:
        with _default_lock:
            if _default_manager is None:
                _default_manager = ModelManager.instance(
                    model_root=model_root, device=device, asr_model=asr_model,
                )
    return _default_manager
