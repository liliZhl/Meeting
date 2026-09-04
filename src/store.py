# -*- coding: utf-8 -*-
"""
历史记录存储层（阶段 B）
========================
数据模型：
    records/<id>/
    ├─ meta.json          # 元信息（标题/时间/时长/来源/状态）
    ├─ audio.wav          # 音频副本（16k 单声道，播放 + 转写对齐）
    ├─ transcript.json    # 结构化转写结果（可选）
    └─ summary.md         # 会议纪要（可选）

records/index.json        # 总索引（按时间倒序）

状态机：
    pending      仅音频，未转写
    transcribed  有转写，无纪要
    summarized   有纪要（隐含已转写）

供 UI 调用，不依赖 PyQt（纯文件层，便于测试）。
"""
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 状态常量
# ---------------------------------------------------------------------------
ST_PENDING = "pending"           # 仅音频
ST_TRANSCRIBED = "transcribed"   # 有转写
ST_SUMMARIZED = "summarized"     # 有纪要

_INDEX_FILE = "index.json"

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _new_id() -> str:
    """记录 id：时间戳 + 随机短串，避免重名冲突。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# RecordStore
# ---------------------------------------------------------------------------

class RecordStore:
    """历史记录管理（目录 + 索引 + 状态机）。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.records_dir = self.root / "records"
        self.index_path = self.records_dir / _INDEX_FILE
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self._index = self._load_index()

    # ---------- 索引 ----------
    def _load_index(self) -> dict:
        return _read_json(self.index_path)

    def _save_index(self) -> None:
        _write_json(self.index_path, self._index)

    def list_records(self) -> list[dict]:
        """返回记录列表（新 -> 旧）。"""
        recs = self._index.get("records", [])
        recs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return recs

    def get(self, rid: str) -> dict | None:
        for r in self.list_records():
            if r.get("id") == rid:
                return r
        return None

    def record_dir(self, rid: str) -> Path:
        return self.records_dir / rid

    def audio_path(self, rid: str) -> Path:
        return self.record_dir(rid) / "audio.wav"

    def transcript_path(self, rid: str) -> Path:
        return self.record_dir(rid) / "transcript.json"

    def summary_path(self, rid: str) -> Path:
        return self.record_dir(rid) / "summary.md"

    # ---------- 创建 ----------
    def create(self, title: str = "", source: str = "recorded",
               duration_sec: float = 0.0) -> str:
        """新建记录（仅音频），返回记录 id。"""
        rid = _new_id()
        (self.records_dir / rid).mkdir(parents=True, exist_ok=True)
        meta = {
            "id": rid,
            "title": title or f"会议 {datetime.now().strftime('%m-%d %H:%M')}",
            "created_at": _now_iso(),
            "duration_sec": round(duration_sec, 2),
            "source": source,          # recorded | imported
            "status": ST_PENDING,
            "asr_model": "",
        }
        self._add_to_index(meta)
        return rid

    def _add_to_index(self, meta: dict) -> None:
        recs = self._index.setdefault("records", [])
        # 同 id 覆盖
        recs = [r for r in recs if r.get("id") != meta["id"]]
        recs.append(meta)
        self._index["records"] = recs
        self._save_index()

    # ---------- 音频 ----------
    def save_audio(self, rid: str, src_wav: str | Path) -> Path:
        """把 16k wav 复制进记录目录，返回目标路径。"""
        dst = self.audio_path(rid)
        shutil.copy2(str(src_wav), str(dst))
        # 更新时长（若无）
        return dst

    # ---------- 转写 ----------
    def save_transcript(self, rid: str, result_obj) -> str:
        """保存结构化转写结果（TranscriptionResult），返回 json 路径。"""
        data = result_obj.to_dict() if hasattr(result_obj, "to_dict") else {
            "sentences": [], "speakers": {}
        }
        data["saved_at"] = _now_iso()
        p = self.transcript_path(rid)
        _write_json(p, data)
        self.set_status(rid, ST_TRANSCRIBED)
        return str(p)

    def load_transcript(self, rid: str) -> dict | None:
        p = self.transcript_path(rid)
        if not p.exists():
            return None
        return _read_json(p)

    # ---------- 纪要 ----------
    def save_summary(self, rid: str, text: str) -> str:
        p = self.summary_path(rid)
        p.write_text(text, encoding="utf-8")
        self.set_status(rid, ST_SUMMARIZED)
        return str(p)

    def load_summary(self, rid: str) -> str:
        p = self.summary_path(rid)
        return p.read_text(encoding="utf-8") if p.exists() else ""

    # ---------- 状态 ----------
    def set_status(self, rid: str, status: str) -> None:
        recs = self._index.setdefault("records", [])
        for r in recs:
            if r.get("id") == rid:
                r["status"] = status
                r.setdefault("updated_at", _now_iso())
                r["updated_at"] = _now_iso()
                break
        self._save_index()

    def status_of(self, rid: str) -> str:
        r = self.get(rid)
        return r.get("status", ST_PENDING) if r else ST_PENDING

    # ---------- 删除 ----------
    def delete(self, rid: str) -> bool:
        """删除记录目录 + 索引条目。"""
        d = self.record_dir(rid)
        if d.exists():
            shutil.rmtree(str(d), ignore_errors=True)
        recs = self._index.setdefault("records", [])
        before = len(recs)
        self._index["records"] = [r for r in recs if r.get("id") != rid]
        if len(self._index["records"]) != before:
            self._save_index()
            return True
        return False

    # ---------- 重命名 ----------
    def rename(self, rid: str, title: str) -> None:
        recs = self._index.setdefault("records", [])
        for r in recs:
            if r.get("id") == rid:
                r["title"] = title
                break
        self._save_index()

    # ---------- 说话人映射持久化 ----------
    def save_speakers(self, rid: str, speakers: dict) -> None:
        """重命名/合并说话人后更新 transcript.json 里的 speakers 映射。"""
        data = self.load_transcript(rid)
        if data is None:
            return
        data["speakers"] = {str(k): v for k, v in speakers.items()}
        # 同时改写句子中的 speaker 编号（合并场景：句子编号已统一）
        _write_json(self.transcript_path(rid), data)

    # ---------- 旧录音迁移（防御性） ----------
    @classmethod
    def migrate_legacy(cls, store: "RecordStore", legacy_dir: str | Path) -> int:
        """扫描旧 recordings/*.wav 转成记录。

        迁移后保留原文件不删除。返回迁移条数。
        幂等：已迁移过的文件名写入 meta 的 legacy_file 字段，跳过。
        """
        ld = Path(legacy_dir)
        if not ld.exists():
            return 0
        migrated = 0
        for wav in sorted(ld.glob("*.wav")):
            title = wav.stem
            # 查是否已迁移
            exists = any(
                r.get("legacy_file") == wav.name
                for r in store.list_records()
            )
            if exists:
                continue
            try:
                rid = store.create(title=title, source="imported")
                meta = store.get(rid)
                if meta is not None:
                    meta["legacy_file"] = wav.name
                    store._add_to_index(meta)
                store.save_audio(rid, wav)
                migrated += 1
            except Exception:
                continue
        return migrated


# ---------------------------------------------------------------------------
# 便捷：自动状态校正（打开记录时按实际文件修正状态）
# ---------------------------------------------------------------------------

def sync_status(store: RecordStore, rid: str) -> str:
    """按实际文件把状态校准为最完整的状态（幂等）。"""
    has_sum = store.summary_path(rid).exists()
    has_tr = store.transcript_path(rid).exists()
    if has_sum:
        st = ST_SUMMARIZED
    elif has_tr:
        st = ST_TRANSCRIBED
    else:
        st = ST_PENDING
    if store.status_of(rid) != st:
        store.set_status(rid, st)
    return st
