# -*- coding: utf-8 -*-
"""
ASR 推理引擎 — 进程内调用 FunASR AutoModel（含说话人分离）

架构说明：
  不使用 funasr-server 子进程，改为进程内直接加载模型推理，
  便于 PyInstaller 打包与分发（拷给别人零依赖）。

模型链路（说话人分离）：
  Fun-ASR-Nano-2512（ASR 识别，LLM 架构 800M）
  + fsmn-vad（语音活动检测，切分长音频）
  + cam++（说话人嵌入，区分谁在说话）
  + ct-punc（标点恢复）

输出：结构化转写结果（句子级：时间戳 + 说话人 + 文本），
      可再格式化为带时间戳与说话人标签的纯文本。

注意：
  - 说话人分离需要从源码安装 FunASR：
      pip install git+https://github.com/modelscope/FunASR.git
  - 分发时模型放本地 mod/ 目录，加载时用本地绝对路径，
    不依赖在线下载。
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger("MeetingAssistant.ASR")

# ---------------------------------------------------------------------------
# 本地模型子目录名（相对 model_root）
# ---------------------------------------------------------------------------
SUB_ASR = "fun-asr-nano"       # 主识别模型（Fun-ASR-Nano-2512）
SUB_VAD = "fsmn-vad"           # 语音活动检测
SUB_SPK = "cam++"              # 说话人嵌入（cam++）
SUB_PUNC = "ct-punc"           # 标点恢复

DEFAULT_MODEL_ROOT = "mod"


# ---------------------------------------------------------------------------
# 结构化转写结果
# ---------------------------------------------------------------------------
class TranscriptionResult:
    """
    结构化转写结果。

    属性：
      sentences: list[dict]，每句含：
          start_ms: int  起始时间（毫秒）
          end_ms:   int  结束时间（毫秒）
          speaker:  int | None  说话人编号（1 起始，None=未标注）
          text:     str  该句文本
      speakers: dict[int, str]  编号 -> 显示名（如 {1: "说话人1"}）

    说明：
      - speaker 编号统一为 1 起始（引擎解析时由 0 起始 +1），
        与 UI 语义一致，便于用户重命名/合并。
      - to_text() 生成与旧版一致的纯文本（供 AI 总结、导出、兼容显示）。
    """

    def __init__(self, sentences=None, speakers=None):
        self.sentences = sentences or []
        self.speakers = speakers or {}

    def to_text(self, use_speaker_names=True) -> str:
        """转成纯文本：每句一行 `[HH:MM:SS-HH:MM:SS] 【说话人N】text`。"""
        lines = []
        for s in self.sentences:
            text = (s.get("text") or "").strip()
            if not text:
                continue
            ts = self._fmt_range(s.get("start_ms"), s.get("end_ms"))
            spk = s.get("speaker")
            if spk is not None and use_speaker_names:
                name = self.speakers.get(spk, f"说话人{spk}")
                lines.append(f"[{ts}] 【{name}】{text}")
            else:
                lines.append(f"[{ts}] {text}")
        if lines:
            return "\n".join(lines)
        # 兜底：无句子但有整体文本
        raw = "".join((s.get("text") or "") for s in self.sentences).strip()
        return raw

    @staticmethod
    def _fmt_range(start_ms, end_ms):
        def _fmt(ms):
            if ms is None:
                return "00:00:00"
            s = int(ms) // 1000
            h, rem = divmod(s, 3600)
            m, sec = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{sec:02d}"

        if start_ms is None and end_ms is None:
            return "00:00:00"
        return f"{_fmt(start_ms)}-{_fmt(end_ms)}"

    def to_dict(self) -> dict:
        """序列化为 dict（供 transcript.json 持久化）。"""
        return {"sentences": self.sentences, "speakers": self.speakers}

    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptionResult":
        d = data or {}
        return cls(
            sentences=[dict(s) for s in d.get("sentences", [])],
            speakers={int(k): v for k, v in (d.get("speakers") or {}).items()},
        )


class ASREngine:
    """
    ASR 推理引擎封装。
    负责模型加载（懒加载）、音频转写、说话人分离、结果结构化。
    注：引擎实例可常驻复用（模型加载一次，多次转写）。
    """

    def __init__(self, model_root: str = DEFAULT_MODEL_ROOT, device: str = None):
        # 归一化模型根目录为绝对路径
        p = Path(model_root)
        if not p.is_absolute():
            # 相对路径：优先相对应用目录，其次相对当前工作目录
            try:
                import sys
                if getattr(sys, "frozen", False):
                    base = Path(sys.executable).parent
                else:
                    base = Path(__file__).resolve().parent.parent
                p = base / model_root
            except Exception:
                p = Path(model_root).resolve()
        self.model_root = str(p)

        self.device = device or self._detect_device()
        self.model = None
        self._loading = False
        logger.info(f"ASREngine 初始化，device={self.device}, model_root={self.model_root}")

    # ---- 路径解析 ----
    def _sub(self, name: str) -> str:
        """返回某模型的本地绝对路径。"""
        return str(Path(self.model_root) / name)

    def _check_local_models(self) -> list:
        """检查本地模型是否齐全，返回缺失列表。"""
        missing = []
        for sub in (SUB_ASR, SUB_VAD, SUB_SPK, SUB_PUNC):
            d = Path(self._sub(sub))
            if not d.exists():
                missing.append(sub)
        return missing

    # ---- 设备检测 ----
    def _detect_device(self) -> str:
        """检测可用设备：GPU -> CPU。"""
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                logger.info(f"检测到 GPU: {name}")
                return "cuda:0"
            logger.info("未检测到 GPU，使用 CPU")
            return "cpu"
        except Exception as e:
            logger.warning(f"GPU 检测失败，使用 CPU: {e}")
            return "cpu"

    # ---- 模型加载 ----
    def load_model(self, progress_callback=None):
        """
        加载模型（懒加载，可被 ModelManager 提前调用）。
        progress_callback: 可选，接收 (step_name:str, detail:str) 更新 UI。
        成功返回 True；失败抛 RuntimeError。
        """
        if self.model is not None:
            logger.info("模型已加载，跳过")
            return True
        if self._loading:
            logger.warning("模型正在加载中")
            return False

        self._loading = True
        try:
            if progress_callback:
                progress_callback("加载模型", "正在加载 Fun-ASR-Nano + 说话人分离模型…")

            # 校验本地模型齐全
            missing = self._check_local_models()
            if missing:
                raise RuntimeError(f"缺少模型文件: {', '.join(missing)}，请检查 mod/ 目录")

            from funasr import AutoModel

            # Fun-ASR-Nano 是 LLM 架构。funasr 1.4.5 已内置其实现，
            # trust_remote_code=True 会自动加载模型目录内的 model.py。
            # 关键：model.py 依赖同目录的 ctc.py 和 tools/（from tools.utils import forced_align），
            # 故需把模型目录加入 sys.path 并切换 cwd，确保依赖可导入。
            asr_dir = self._sub(SUB_ASR)
            import sys
            if asr_dir not in sys.path:
                sys.path.insert(0, asr_dir)
            os.chdir(asr_dir)

            logger.info("开始加载 FunASR AutoModel（本地路径 + VAD + cam++ + punc）")
            self.model = AutoModel(
                model=asr_dir,                  # 本地主模型路径
                trust_remote_code=True,
                vad_model=self._sub(SUB_VAD),   # 本地 VAD
                vad_kwargs={"max_single_segment_time": 30000},
                spk_model=self._sub(SUB_SPK),   # 本地说话人嵌入（cam++）
                device=self.device,
                disable_update=True,            # 关闭版本检查，避免联网
            )
            logger.info("模型加载完成")
            return True
        except Exception as e:
            logger.exception("模型加载失败")
            raise RuntimeError(f"模型加载失败：{e}") from e
        finally:
            self._loading = False

    # ---- 转写 ----
    def transcribe(self, audio_path: str, progress_callback=None) -> TranscriptionResult:
        """
        转写音频，返回结构化 TranscriptionResult（句子级时间戳 + 说话人）。

        会先把任意音频（mp3/m4a/flac 等）统一转成 16kHz 单声道 wav，
        避免 funasr 内部因编码/中文路径/特殊 mp3 编码加载失败。

        progress_callback: 可选，接收 (step_name:str, detail:str)。
        """
        if self.model is None:
            self.load_model(progress_callback=progress_callback)

        # 统一转成 16kHz 单声道 wav，保证加载稳定
        wav_path = self._ensure_wav_16k(audio_path, progress_callback)

        if progress_callback:
            progress_callback("转写中", f"正在识别「{Path(audio_path).name}」…")

        logger.info(f"开始转写: {audio_path} (实际转写文件: {wav_path})")
        res = self.model.generate(
            input=[wav_path],
            cache={},
            batch_size=1,
            language="中文",
            itn=True,
        )

        result = self._parse_result(res)
        logger.info(f"转写完成: {len(result.sentences)} 句")
        return result

    def _ensure_wav_16k(self, audio_path: str, progress_callback=None) -> str:
        """确保音频为 16kHz 单声道 wav；必要时用 ffmpeg 转换并返回临时路径。"""
        p = Path(audio_path)
        # 已是标准 wav 则直接返回
        if p.suffix.lower() == ".wav":
            try:
                import wave
                with wave.open(str(p), "rb") as wf:
                    if wf.getframerate() == 16000 and wf.getnchannels() == 1:
                        return str(p)
            except Exception:
                pass

        # 查找 ffmpeg（优先应用 bin/ 下的，其次 PATH）
        ffmpeg = self._find_ffmpeg()
        if not ffmpeg:
            # 无 ffmpeg 时只能原样交给 funasr 尝试加载
            return audio_path

        if progress_callback:
            progress_callback("转换音频", f"正在把「{p.name}」转成 16kHz wav…")

        tmp = Path(tempfile.gettempdir()) / f"asr_conv_{os.getpid()}_{int(__import__('time').time()*1000)}.wav"
        cmd = [ffmpeg, "-y", "-i", str(p),
               "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(tmp)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"音频已转换为 wav: {tmp}")
            return str(tmp)
        except Exception as e:
            logger.warning(f"ffmpeg 转换失败，原样加载: {e}")
            return audio_path

    def _find_ffmpeg(self):
        """查找 ffmpeg 可执行文件：应用 bin/ 目录 -> PATH。"""
        candidates = []
        try:
            import sys
            if getattr(sys, "frozen", False):
                candidates.append(str(Path(sys.executable).parent / "bin" / "ffmpeg.exe"))
            else:
                candidates.append(str(Path(__file__).resolve().parent.parent / "bin" / "ffmpeg.exe"))
        except Exception:
            pass
        for c in candidates:
            if c and os.path.exists(c):
                return c
        # 回退：PATH 里的 ffmpeg
        found = shutil.which("ffmpeg")
        return found

    # ---- 结果解析（结构化） ----
    def _parse_result(self, res) -> TranscriptionResult:
        """
        解析 AutoModel 返回结果，构建结构化 TranscriptionResult。

        说明：
          - Fun-ASR-Nano 的 sentence_info 每条字段为：
              start/end(ms)、sentence(文本)、timestamp(字级)、spk(数字, 0 起始)
          - 解析时 spk 统一 +1 转成 1 起始（UI 语义一致）。
        """
        sentences = []
        speakers = {}
        try:
            if isinstance(res, list) and len(res) > 0:
                first = res[0]
                sentence_info = first.get("sentence_info", [])
                if sentence_info:
                    for sent in sentence_info:
                        text = (sent.get("sentence") or sent.get("text") or "").strip()
                        if not text:
                            continue
                        spk0 = sent.get("spk")
                        spk = None
                        if spk0 is not None:
                            try:
                                spk = int(spk0) + 1   # 0 起始 -> 1 起始
                            except (TypeError, ValueError):
                                spk = None
                        if spk is not None and spk not in speakers:
                            speakers[spk] = f"说话人{spk}"
                        sentences.append({
                            "start_ms": sent.get("start"),
                            "end_ms": sent.get("end"),
                            "speaker": spk,
                            "text": text,
                        })
                    return TranscriptionResult(sentences, speakers)
                # 无 sentence_info：整体文本兜底为单句
                plain = (first.get("text") or "").strip()
                if plain:
                    sentences.append({
                        "start_ms": None, "end_ms": None,
                        "speaker": None, "text": plain,
                    })
                    return TranscriptionResult(sentences, speakers)
        except Exception as e:
            logger.warning(f"解析转写结果失败: {e}")

        if isinstance(res, list) and res:
            raw = str(res[0].get("text", ""))
            if raw.strip():
                sentences.append({
                    "start_ms": None, "end_ms": None,
                    "speaker": None, "text": raw.strip(),
                })
        return TranscriptionResult(sentences, speakers)


def transcribe_audio(audio_path: str, model_root: str = DEFAULT_MODEL_ROOT,
                     device: str = None, progress_callback=None) -> TranscriptionResult:
    """一次性转写（创建引擎 -> 加载 -> 转写），返回结构化结果。"""
    engine = ASREngine(model_root=model_root, device=device)
    engine.load_model(progress_callback=progress_callback)
    return engine.transcribe(audio_path, progress_callback=progress_callback)
