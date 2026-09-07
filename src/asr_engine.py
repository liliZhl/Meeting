# -*- coding: utf-8 -*-
"""
ASR 推理引擎 — 进程内调用 FunASR AutoModel（含说话人分离）

架构说明：
  不使用 funasr-server 子进程，改为进程内直接加载模型推理，
  便于 PyInstaller 打包与分发（拷给别人零依赖）。

模型链路（说话人分离）：
  主识别模型可选（用户在配置中自选）：
    - Fun-ASR-Nano-2512（LLM 架构 800M，GPU/强 CPU，质量最高）
    - SenseVoice-Small（~1GB，CPU 可实时，多语种，带情感检测）
    - Paraformer-large（~1GB，中文最准，字级时间戳）
  公共组件：
    + fsmn-vad（语音活动检测，切分长音频）
    + cam++（说话人嵌入，区分谁在说话）
    + ct-punc（标点恢复，仅 Paraformer 需要）

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


def clean_asr_text(text: str) -> str:
    """清理 ASR 输出的富文本标签与空白。

    SenseVoice 输出形如 '<|zh|><|HAPPY|><|Speech|><|withitn|>文本'，
    带语言/情感/事件/文本规范化标签；funasr 只在部分路径（VAD 拼接）
    剥离，直接透传时仍会残留，故解析层统一清理。对其它模型无副作用。
    """
    import re
    if not text:
        return ""
    text = re.sub(r"<\|[^|]*\|>", "", text)
    return text.strip()

# ---------------------------------------------------------------------------
# 本地模型子目录名（相对 model_root）
# ---------------------------------------------------------------------------
SUB_VAD = "fsmn-vad"           # 语音活动检测（公共组件）
SUB_SPK = "cam++"              # 说话人嵌入（cam++，公共组件）
SUB_PUNC = "ct-punc"           # 标点恢复（Nano/SenseVoice 自带标点时可不用）

# 可选主识别模型（用户在配置里自选，适应有无 GPU 的机器）
MODEL_NANO = "fun-asr-nano"      # Fun-ASR-Nano-2512：LLM 架构 2GB，GPU 旗舰（中英日+方言）
MODEL_SENSEVOICE = "sensevoice"  # SenseVoice-Small ~1GB：CPU 可实时，50+ 语种，带情感/事件
MODEL_PARAFORMER = "paraformer"  # Paraformer-large ~1GB：中文最准之一，CPU 较快

# 模型注册表：key -> 配置
#   dir:       mod/ 下的子目录名
#   label:     UI 显示名
#   desc:      一句话说明（CPU/GPU 适用性）
#   llm:       True=LLM 架构（需 llm_kwargs 抑制重复幻觉；自带标点）
#   vad_kwargs: 加载时透传的 VAD 参数
#   gen_extra: generate 时附加的 kwargs
ASR_MODELS = {
    MODEL_NANO: {
        "label": "Fun-ASR-Nano（高精度 · 需 GPU 或强 CPU）",
        "desc": "LLM 架构 800M，中英日+方言+口音，识别质量最高；CPU 上慢（约 0.5x 实时）",
        "llm": True,
        "gen_extra": {
            "language": "中文",
            "itn": True,
            # 抑制 LLM 解码重复幻觉（"幺幺幺…""包子包子…"）——2026-09-02 实测
            "llm_kwargs": {"repetition_penalty": 1.15, "no_repeat_ngram_size": 3},
        },
    },
    MODEL_SENSEVOICE: {
        "label": "SenseVoice-Small（CPU 首选 · 快速）",
        "desc": "~1GB，非自回归，CPU 可实时；50+ 语种（中英日韩粤等），自带标点与情感检测",
        "llm": False,
        "gen_extra": {
            "language": "auto",
            "use_itn": True,
            "ban_emo_unk": True,   # 不输出 <|HAPPY|> 等情感标签
        },
    },
    MODEL_PARAFORMER: {
        "label": "Paraformer-large（中文会议 · 字级时间戳）",
        "desc": "~1GB，纯中文最准之一，CTC 输出字级精确时间戳；CPU 较快",
        "llm": False,
        "gen_extra": {
            "language": "zh",
            # 注意：batch_size 由 transcribe() 统一传入，这里不重复设置
            # pred_timestamp: Paraformer 输出字级时间戳的必要开关，
            #   缺省时无 timestamp，说话人分离/切句会退化为 VAD 粒度
            "pred_timestamp": True,
            "sentence_timestamp": True,
        },
    },
}

DEFAULT_MODEL_ROOT = "mod"
DEFAULT_ASR_MODEL = MODEL_NANO   # 默认仍用 Nano（有 GPU 目标机）

# ---------------------------------------------------------------------------
# VAD 切句灵敏度预设（fsmn-vad 的 max_end_silence_time 参数）
# ---------------------------------------------------------------------------
# max_end_silence_time: 段尾静音超过该毫秒数才判定说话结束并切段。
#   值越小切得越碎（每句短），越大越粗（长句/可能混说话人）。
#   2026-09-02 实测（标准录音 7.mp3，4分45秒）：
#     800ms -> 105 段（41 段 <1s，过碎）
#     1500ms -> 29 段（平衡）
#     2500ms -> 16 段（完整但可能合并不同说话人）
VAD_LEVEL_FINE = "fine"        # 细：800ms（同模型默认，切句短促）
VAD_LEVEL_MEDIUM = "medium"    # 中：1500ms（推荐，会议平衡）
VAD_LEVEL_COARSE = "coarse"    # 粗：2500ms（长句优先）

VAD_PRESETS = {
    VAD_LEVEL_FINE: {
        "label": "细（停顿 0.8s 即切句）",
        "max_end_silence_time": 800,
    },
    VAD_LEVEL_MEDIUM: {
        "label": "中（停顿 1.5s 才切句，推荐）",
        "max_end_silence_time": 1500,
    },
    VAD_LEVEL_COARSE: {
        "label": "粗（停顿 2.5s 才切句）",
        "max_end_silence_time": 2500,
    },
}

DEFAULT_VAD_LEVEL = VAD_LEVEL_MEDIUM   # 默认中档 1.5s


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

    def __init__(self, model_root: str = DEFAULT_MODEL_ROOT, device: str = None,
                 asr_model: str = None, vad_level: str = None):
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

        # 主识别模型选择（默认 fun-asr-nano）
        self.asr_model = asr_model or DEFAULT_ASR_MODEL
        if self.asr_model not in ASR_MODELS:
            logger.warning(f"未知的 ASR 模型 '{self.asr_model}'，回退到 {DEFAULT_ASR_MODEL}")
            self.asr_model = DEFAULT_ASR_MODEL
        self._model_cfg = ASR_MODELS[self.asr_model]

        # VAD 切句灵敏度（细/中/粗 -> max_end_silence_time）
        self.vad_level = vad_level or DEFAULT_VAD_LEVEL
        if self.vad_level not in VAD_PRESETS:
            logger.warning(f"未知的 VAD 档位 '{self.vad_level}'，回退到 {DEFAULT_VAD_LEVEL}")
            self.vad_level = DEFAULT_VAD_LEVEL
        self._vad_preset = VAD_PRESETS[self.vad_level]

        self.device = device or self._detect_device()
        self.model = None
        self._loading = False
        logger.info(f"ASREngine 初始化，device={self.device}, model_root={self.model_root}, "
                    f"asr_model={self.asr_model}, vad_level={self.vad_level}")

    # ---- 路径解析 ----
    def _sub(self, name: str) -> str:
        """返回某模型的本地绝对路径。"""
        return str(Path(self.model_root) / name)

    def _check_local_models(self) -> list:
        """检查本地模型是否齐全，返回缺失列表。

        主模型按当前所选 asr_model 检查；公共组件（vad/spk）始终需要；
        ct-punc 仅 Paraformer 需要（Nano/SenseVoice 自带标点）。
        """
        missing = []
        # 主识别模型
        if not Path(self._sub(self.asr_model)).exists():
            missing.append(self.asr_model)
        # 公共组件
        for sub in (SUB_VAD, SUB_SPK):
            if not Path(self._sub(sub)).exists():
                missing.append(sub)
        # Paraformer 需要 ct-punc（切句/标点），Nano/SenseVoice 自带
        if self.asr_model == MODEL_PARAFORMER and not Path(self._sub(SUB_PUNC)).exists():
            missing.append(SUB_PUNC)
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
                progress_callback("加载模型",
                                  f"正在加载 {self._model_cfg['label']} + 说话人分离模型…")

            # 校验本地模型齐全
            missing = self._check_local_models()
            if missing:
                raise RuntimeError(f"缺少模型文件: {', '.join(missing)}，请检查 mod/ 目录")

            from funasr import AutoModel

            asr_dir = self._sub(self.asr_model)

            # Fun-ASR-Nano 是 LLM 架构。funasr 1.4.5 已内置其实现，
            # trust_remote_code=True 会自动加载模型目录内的 model.py。
            # 关键：model.py 依赖同目录的 ctc.py 和 tools/（from tools.utils import forced_align），
            # 故需把模型目录加入 sys.path 并切换 cwd，确保依赖可导入。
            # （SenseVoice/Paraformer 不依赖本地 model.py，切换无副作用。）
            import sys
            if asr_dir not in sys.path:
                sys.path.insert(0, asr_dir)
            os.chdir(asr_dir)

            # 通用加载参数：本地主模型 + VAD + 说话人（+ Paraformer 用 ct-punc）
            # vad_kwargs: max_single_segment_time 防超长段（>30s 硬切，
            #   避免单段过长；max_end_silence_time 按用户选的灵敏度档位）
            vad_kwargs = {"max_single_segment_time": 30000,
                          "max_end_silence_time": self._vad_preset["max_end_silence_time"]}
            load_kwargs = dict(
                model=asr_dir,                  # 本地主模型路径
                trust_remote_code=True,
                vad_model=self._sub(SUB_VAD),   # 本地 VAD
                vad_kwargs=vad_kwargs,
                spk_model=self._sub(SUB_SPK),   # 本地说话人嵌入（cam++）
                device=self.device,
                disable_update=True,            # 关闭版本检查，避免联网
            )
            if self.asr_model == MODEL_PARAFORMER:
                # Paraformer 不带标点，需 ct-punc 参与分句/说话人切分
                load_kwargs["punc_model"] = self._sub(SUB_PUNC)

            logger.info(f"开始加载 FunASR AutoModel（{self.asr_model} + VAD + cam++）")
            self.model = AutoModel(**load_kwargs)
            logger.info("模型加载完成")
            return True
        except Exception as e:
            logger.exception("模型加载失败")
            raise RuntimeError(f"模型加载失败：{e}") from e
        finally:
            self._loading = False

    # ---- 转写 ----
    def transcribe(self, audio_path: str, progress_callback=None,
                   num_speakers: int = None) -> TranscriptionResult:
        """
        转写音频，返回结构化 TranscriptionResult（句子级时间戳 + 说话人）。

        会先把任意音频（mp3/m4a/flac 等）统一转成 16kHz 单声道 wav，
        避免 funasr 内部因编码/中文路径/特殊 mp3 编码加载失败。

        num_speakers: 预设说话人数（2..15）。None/<=1 = 自动估计（谱聚类默认）。
            2026-09-07 路线C：funasr 定制版 ClusterBackend 谱聚类支持 1..15 人，
            显式传入时强制按 N 人聚类（oracle_num），适用于 3 人+ 会议。
        """
        if self.model is None:
            self.load_model(progress_callback=progress_callback)

        # 统一转成 16kHz 单声道 wav，保证加载稳定
        wav_path = self._ensure_wav_16k(audio_path, progress_callback)

        if progress_callback:
            progress_callback("正在识别", "")

        logger.info(f"开始转写: {audio_path} (实际转写文件: {wav_path}, 模型: {self.asr_model})")
        # 模型专属 generate 参数（Nano 的 llm_kwargs 抑制重复幻觉等）
        gen_kwargs = dict(self._model_cfg["gen_extra"])
        if num_speakers and num_speakers > 1:
            gen_kwargs["preset_spk_num"] = int(num_speakers)
            logger.info(f"说话人聚类: 预设 {num_speakers} 人（preset_spk_num）")
        res = self.model.generate(
            input=[wav_path],
            cache={},
            batch_size=1,
            **gen_kwargs,
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
            progress_callback("转换音频", "正在把音频转成 16kHz wav…")

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
                        text = clean_asr_text(sent.get("sentence") or sent.get("text") or "")
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
                plain = clean_asr_text(first.get("text") or "")
                if plain:
                    sentences.append({
                        "start_ms": None, "end_ms": None,
                        "speaker": None, "text": plain,
                    })
                    return TranscriptionResult(sentences, speakers)
        except Exception as e:
            logger.warning(f"解析转写结果失败: {e}")

        if isinstance(res, list) and res:
            raw = clean_asr_text(str(res[0].get("text", "")))
            if raw:
                sentences.append({
                    "start_ms": None, "end_ms": None,
                    "speaker": None, "text": raw.strip(),
                })
        return TranscriptionResult(sentences, speakers)


def transcribe_audio(audio_path: str, model_root: str = DEFAULT_MODEL_ROOT,
                     device: str = None, asr_model: str = None,
                     progress_callback=None, num_speakers: int = None) -> TranscriptionResult:
    """一次性转写（创建引擎 -> 加载 -> 转写），返回结构化结果。"""
    engine = ASREngine(model_root=model_root, device=device, asr_model=asr_model)
    engine.load_model(progress_callback=progress_callback)
    return engine.transcribe(audio_path, progress_callback=progress_callback,
                             num_speakers=num_speakers)
