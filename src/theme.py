# -*- coding: utf-8 -*-
"""
theme.py — 集中式主题（阶段 E）
===============================

深色蓝紫（Tokyo Night 风格）+ 浅色两套配色，纯 QSS 实现，不引入第三方 UI 库。
所有控件样式集中在 build_qss() 生成，避免散落在业务代码里互相打架。

用法：
    from theme import build_qss, get_colors
    app.setStyleSheet(build_qss("dark"))     # dark | light
    colors = get_colors("dark")              # 富文本/画布用色
"""

# ---------------------------------------------------------------------------
# 配色表
# ---------------------------------------------------------------------------
DARK = {
    "bg": "#1a1b26",          # 主背景
    "panel": "#24283b",       # 面板背景
    "panel_hover": "#2f3349", # 面板/按钮 hover
    "input": "#1f2335",       # 输入框/文本框背景
    "text": "#c0caf5",        # 文本主色
    "text_dim": "#7d87b0",    # 次要文本（提示、说明）
    "accent": "#7aa2f7",      # 强调色（蓝紫）
    "accent_hover": "#8fb3fa",
    "accent_pressed": "#5f86e0",
    "success": "#9ece6a",     # 就绪/成功
    "danger": "#f7768e",      # 录音/危险操作
    "warn": "#e0af68",
    "border": "#3b4261",      # 边框
    "sel_bg": "#33467c",      # 列表选中背景
    "sel_text": "#ffffff",
    "scroll_handle": "#414868",
    # 富文本（转写区 HTML）用色
    "ts": "#7aa2f7",          # 时间戳
    "spk": "#9ece6a",         # 说话人
    "body": "#c0caf5",        # 正文
    "highlight": "255,200,60,70",   # 播放高亮 rgba
}

LIGHT = {
    "bg": "#f4f5fa",
    "panel": "#ffffff",
    "panel_hover": "#eceefb",
    "input": "#ffffff",
    "text": "#2a2f4a",
    "text_dim": "#6b7093",
    "accent": "#3d6ddb",
    "accent_hover": "#5a86e8",
    "accent_pressed": "#2f57b0",
    "success": "#3f9e52",
    "danger": "#d94a5f",
    "warn": "#b8801a",
    "border": "#d7dae8",
    "sel_bg": "#d9e2fb",
    "sel_text": "#1a1b26",
    "scroll_handle": "#c2c8e0",
    "ts": "#3d6ddb",
    "spk": "#3f9e52",
    "body": "#2a2f4a",
    "highlight": "255,214,10,110",
}

THEMES = {"dark": DARK, "light": LIGHT}


def get_colors(mode: str = "dark") -> dict:
    """返回配色字典（未知模式回退到 dark）。"""
    return THEMES.get(mode, DARK)


# ---------------------------------------------------------------------------
# QSS 生成
# ---------------------------------------------------------------------------

def build_qss(mode: str = "dark") -> str:
    """生成整套 QSS 字符串。"""
    c = get_colors(mode)
    return f"""
/* ===== 全局 ===== */
QWidget {{
    background-color: {c['bg']};
    color: {c['text']};
    font-family: 'Microsoft YaHei', 'Segoe UI';
    font-size: 13px;
}}
QMainWindow, QDialog, QMessageBox, QInputDialog {{
    background-color: {c['bg']};
}}
QToolTip {{
    background-color: {c['panel']};
    color: {c['text']};
    border: 1px solid {c['border']};
    padding: 4px 6px;
    border-radius: 4px;
}}

/* ===== 标签 ===== */
QLabel {{ color: {c['text']}; background: transparent; }}
QLabel#titleLabel {{
    font-size: 20px; font-weight: bold; color: {c['accent']};
    padding: 2px 0 6px 0; background: transparent;
}}
QLabel#histTitle {{
    font-weight: bold; color: {c['text']}; padding: 4px 2px; background: transparent;
}}
QLabel#lblTranscriptTip, QLabel#lblAudio, QLabel#lblDuration,
QLabel#lblPlayTime, QLabel#tipLabel, QLabel#asrModelDesc {{
    color: {c['text_dim']}; font-size: 12px; background: transparent;
}}

/* ===== 按钮 ===== */
QPushButton {{
    background-color: {c['panel']};
    color: {c['text']};
    border: 1px solid {c['border']};
    border-radius: 6px;
    padding: 6px 16px;
    min-height: 18px;
}}
QPushButton:hover {{ background-color: {c['panel_hover']}; border-color: {c['accent']}; }}
QPushButton:pressed {{ background-color: {c['accent_pressed']}; color: #ffffff; }}
QPushButton:disabled {{
    background-color: {c['panel']};
    color: {c['text_dim']};
    border-color: {c['border']};
}}
/* 主操作：转写 / 生成纪要 */
QPushButton#btnTranscribe, QPushButton#btnSummarize {{
    background-color: {c['accent']};
    color: #ffffff; border: 1px solid {c['accent']}; font-weight: bold;
}}
QPushButton#btnTranscribe:hover, QPushButton#btnSummarize:hover {{
    background-color: {c['accent_hover']};
}}
QPushButton#btnTranscribe:disabled, QPushButton#btnSummarize:disabled {{
    background-color: {c['panel']}; color: {c['text_dim']}; border-color: {c['border']};
}}
/* 录音：危险色 */
QPushButton#btnRecord {{
    background-color: {c['danger']}; color: #ffffff; border: 1px solid {c['danger']};
}}
QPushButton#btnRecord:hover {{ background-color: {c['danger']}; }}
QPushButton#btnRecord:disabled {{
    background-color: {c['panel']}; color: {c['text_dim']}; border-color: {c['border']};
}}
/* 播放：成功色 */
QPushButton#btnPlay {{
    background-color: {c['success']}; color: #1a1b26; border: 1px solid {c['success']};
    min-width: 76px;
}}
QPushButton#btnPlay:hover {{ background-color: {c['success']}; }}
QPushButton#btnPlay:disabled {{
    background-color: {c['panel']}; color: {c['text_dim']}; border-color: {c['border']};
}}
/* 模型加载 */
QPushButton#btnLoadModel {{
    border: 1px dashed {c['accent']}; color: {c['accent']};
}}
QPushButton#btnLoadModel:disabled {{
    border: 1px solid {c['border']}; color: {c['text_dim']}; background-color: {c['panel']};
}}

/* ===== 输入与文本区 ===== */
QLineEdit, QPlainTextEdit, QTextBrowser, QTextEdit {{
    background-color: {c['input']};
    color: {c['text']};
    border: 1px solid {c['border']};
    border-radius: 6px;
    padding: 6px;
    selection-background-color: {c['sel_bg']};
    selection-color: {c['sel_text']};
}}
QPlainTextEdit, QTextBrowser {{
    font-family: 'Consolas', 'Microsoft YaHei';
    font-size: 13px;
    line-height: 1.4;
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextBrowser:focus {{
    border-color: {c['accent']};
}}
QComboBox {{
    background-color: {c['input']}; color: {c['text']};
    border: 1px solid {c['border']}; border-radius: 6px;
    padding: 5px 8px; min-height: 18px;
}}
QComboBox:hover {{ border-color: {c['accent']}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background-color: {c['panel']}; color: {c['text']};
    border: 1px solid {c['border']}; border-radius: 4px;
    selection-background-color: {c['sel_bg']}; selection-color: {c['sel_text']};
}}

/* ===== 历史记录列表 ===== */
QListWidget {{
    background-color: {c['panel']};
    border: 1px solid {c['border']};
    border-radius: 6px;
    padding: 4px;
    outline: none;
}}
QListWidget::item {{
    padding: 7px 8px; border-radius: 4px; margin: 1px 0;
}}
QListWidget::item:hover {{ background-color: {c['panel_hover']}; }}
QListWidget::item:selected {{ background-color: {c['sel_bg']}; color: {c['sel_text']}; }}

/* ===== 面板 ===== */
QWidget#histPanel, QWidget#playBar {{
    background-color: {c['panel']};
    border: 1px solid {c['border']};
    border-radius: 8px;
}}
QWidget#playBar {{ padding: 2px; }}

/* ===== 分割条 ===== */
QSplitter::handle {{ background-color: {c['border']}; }}
QSplitter::handle:hover {{ background-color: {c['accent']}; }}

/* ===== 播放进度滑块 ===== */
QSlider::groove:horizontal {{
    height: 6px; background: {c['border']}; border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {c['accent']}; border-radius: 3px;
}}
QSlider::add-page:horizontal {{
    background: {c['border']}; border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 14px; margin: -5px 0; background: {c['accent']};
    border: 1px solid {c['accent']}; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: {c['accent_hover']}; }}
QSlider::handle:horizontal:disabled {{ background: {c['border']}; }}

/* ===== 进度条 ===== */
QProgressBar {{
    background-color: {c['panel']}; border: 1px solid {c['border']};
    border-radius: 6px; text-align: center; color: {c['text']}; min-height: 16px;
}}
QProgressBar::chunk {{ background-color: {c['accent']}; border-radius: 5px; }}

/* ===== 状态栏 ===== */
QStatusBar {{
    background-color: {c['panel']}; color: {c['text_dim']};
    border-top: 1px solid {c['border']};
}}
QStatusBar QLabel {{ color: {c['text_dim']}; background: transparent; }}

/* ===== 菜单 ===== */
QMenu {{
    background-color: {c['panel']}; color: {c['text']};
    border: 1px solid {c['border']}; border-radius: 6px; padding: 4px;
}}
QMenu::item {{ padding: 6px 22px 6px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background-color: {c['sel_bg']}; color: {c['sel_text']}; }}
QMenu::separator {{ height: 1px; background: {c['border']}; margin: 4px 8px; }}

/* ===== 滚动条 ===== */
QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 0 2px 0 0;
}}
QScrollBar:horizontal {{
    background: transparent; height: 10px; margin: 0 0 2px 0;
}}
QScrollBar::handle:vertical {{
    background: {c['scroll_handle']}; border-radius: 5px; min-height: 24px;
}}
QScrollBar::handle:horizontal {{
    background: {c['scroll_handle']}; border-radius: 5px; min-width: 24px;
}}
QScrollBar::handle:hover {{ background: {c['accent']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
"""
