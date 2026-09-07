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

# 浅色 = 新拟态（Neumorphism 近似，2026-09-07）
# QSS 无 box-shadow，真双层投影做不了，采用「同底色 + 双色渐变描边模拟凹凸 + 大圆角」近似。
LIGHT = {
    "bg": "#e2e7ee",          # 主背景（新拟态米灰底）
    "panel": "#e8ecf3",       # 面板背景（接近底色，靠渐变/描边区分层次）
    "panel_hover": "#dfe5ee",
    "input": "#dfe5ee",       # 输入框/文本区（比底色略深，呈内凹感）
    "text": "#3a4356",
    "text_dim": "#7b849a",
    "accent": "#4c7df0",
    "accent_hover": "#6a92f5",
    "accent_pressed": "#3a66d8",
    "success": "#43a468",
    "danger": "#e05555",
    "warn": "#d7922f",
    "border": "#cdd5e2",      # 浅描边
    "sel_bg": "#4c7df0",
    "sel_text": "#ffffff",
    "scroll_handle": "#c7cfdd",
    "ts": "#3d6ddb",
    "spk": "#3f9e52",
    "body": "#3a4356",
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
    _qss = f"""
/* ===== 全局 ===== */
QWidget {{
    background-color: {c['bg']};
    color: {c['text']};
    font-family: 'Microsoft YaHei', 'Segoe UI';
    font-size: 13px;
}}
/* 主窗口无边框（2026-09-07）：QMainWindow 透明，圆角由 #appRoot 承载 */
QMainWindow {{
    background-color: transparent;
}}
QDialog, QMessageBox, QInputDialog {{
    background-color: {c['bg']};
}}
/* 主窗口圆角卡片 */
QWidget#appRoot {{
    background-color: {c['bg']};
    border-radius: 14px;
}}
/* 无边框弹窗统一卡片（2026-09-07：弹窗全部无边框圆角） */
QDialog#nfDlg {{
    background-color: transparent;
}}
QWidget#dlgCard {{
    background-color: {c['panel']};
    border: 1px solid {c['border']};
    border-radius: 14px;
}}
QLabel#dlgTitle {{
    color: {c['text']};
    font-size: 15px;
    font-weight: bold;
    background: transparent;
    padding: 4px 0 0 2px;
}}
QWidget#dlgTitleBar {{
    background: transparent;
}}
QPushButton#dlgClose {{
    background: transparent;
    border: none;
    border-radius: 6px;
    color: {c['text_dim']};
    font-size: 14px;
    min-width: 26px;
    min-height: 22px;
    padding: 0; margin: 0;
}}
QPushButton#dlgClose:hover {{
    background-color: {c['danger']};
    color: #ffffff;
}}
/* 消息/输入弹窗文本 */
QLabel#dlgText {{
    color: {c['text']};
    font-size: 13px;
    background: transparent;
}}
QPushButton#dlgBtnOK {{
    background-color: {c['accent']};
    color: #ffffff;
    border: 1px solid {c['accent']};
    border-radius: 8px;
    padding: 5px 20px;
}}
QPushButton#dlgBtnOK:hover {{ background-color: {c['accent_hover']}; }}
QPushButton#dlgBtnCancel {{
    background-color: {c['panel']};
    color: {c['text']};
    border: 1px solid {c['border']};
    border-radius: 8px;
    padding: 5px 20px;
}}
QPushButton#dlgBtnCancel:hover {{ background-color: {c['panel_hover']}; border-color: {c['accent']}; }}
QPushButton#dlgBtnDanger {{
    background-color: {c['danger']};
    color: #ffffff;
    border: 1px solid {c['danger']};
    border-radius: 8px;
    padding: 5px 20px;
}}
QPushButton#dlgBtnDanger:hover {{ background-color: {c['danger']}; }}
/* 无边框标题栏（透明，作为拖动区） */
QWidget#titleBar {{ background: transparent; }}
QPushButton#btnWinMin, QPushButton#btnWinMax, QPushButton#btnWinClose {{
    background: transparent;
    border: none;
    border-radius: 6px;
    color: {c['text_dim']};
    font-size: 13px;
    padding: 0; margin: 0;
}}
QPushButton#btnWinMin:hover, QPushButton#btnWinMax:hover {{
    background-color: {c['panel_hover']};
    color: {c['text']};
}}
QPushButton#btnWinClose:hover {{
    background-color: {c['danger']};
    color: #ffffff;
}}
QToolTip {{
    background-color: {c['panel']};
    color: {c['text']};
    border: 1px solid {c['border']};
    padding: 4px 6px;
    border-radius: 6px;
}}

/* ===== 标签 ===== */
QLabel {{ color: {c['text']}; background: transparent; }}
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
    border-radius: 8px;
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
    border-radius: 8px;
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
    border: 1px solid {c['border']}; border-radius: 8px;
    padding: 5px 8px; min-height: 18px;
}}
QComboBox:hover {{ border-color: {c['accent']}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background-color: {c['panel']}; color: {c['text']};
    border: 1px solid {c['border']}; border-radius: 6px;
    selection-background-color: {c['sel_bg']}; selection-color: {c['sel_text']};
}}

/* ===== 历史记录列表 ===== */
QListWidget {{
    background-color: {c['panel']};
    border: 1px solid {c['border']};
    border-radius: 8px;
    padding: 4px;
    outline: none;
}}
QListWidget::item {{
    padding: 7px 8px; border-radius: 6px; margin: 1px 0;
}}
QListWidget::item:hover {{ background-color: {c['panel_hover']}; }}
QListWidget::item:selected {{ background-color: {c['sel_bg']}; color: {c['sel_text']}; }}

/* ===== 面板 ===== */
QWidget#histPanel, QWidget#playBar {{
    background-color: {c['panel']};
    border: 1px solid {c['border']};
    border-radius: 10px;
}}
QWidget#playBar {{ padding: 2px; }}

/* ===== 分割条 ===== */
QSplitter::handle {{ background-color: {c['border']}; }}
QSplitter::handle:hover {{ background-color: {c['accent']}; }}

/* ===== 播放进度滑块（2026-09-07：圆钮视觉全透明，保留可抓热区） ===== */
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
    width: 14px; margin: -5px 0;
    background: transparent;
    border: 1px solid transparent; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: transparent; }}
QSlider::handle:horizontal:disabled {{ background: transparent; }}

/* ===== 进度条 ===== */
QProgressBar {{
    background-color: {c['panel']}; border: 1px solid {c['border']};
    border-radius: 6px; text-align: center; color: {c['text']}; min-height: 16px;
}}
QProgressBar::chunk {{ background-color: {c['accent']}; border-radius: 5px; }}

/* ===== 状态栏（无边框后为普通圆角容器） ===== */
QWidget#statusBar {{
    background-color: {c['panel']};
    border: 1px solid {c['border']};
    border-radius: 9px;
}}
QWidget#statusBar QLabel {{ color: {c['text_dim']}; background: transparent; }}
QWidget#statusBar QLabel#lblStatus {{ color: {c['text']}; }}

/* ===== 菜单 ===== */
QMenu {{
    background-color: {c['panel']}; color: {c['text']};
    border: 1px solid {c['border']}; border-radius: 8px; padding: 4px;
}}
QMenu::item {{ padding: 6px 22px 6px 12px; border-radius: 6px; }}
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
    if mode == "light":
        _qss += _NEO_QSS
    return _qss


# ---------------------------------------------------------------------------
# 新拟态（Neumorphism）浅色专属样式 —— 追加在通用 QSS 之后覆盖
# 近似实现：同色底 + 渐变模拟凸起/凹陷 + 大圆角 + 柔和描边（QSS 无 box-shadow）
# ---------------------------------------------------------------------------
_NEO_QSS = """
/* ===== 面板：柔和的悬浮卡片（凸） ===== */
QWidget#histPanel, QWidget#playBar {
    background-color: #e6eaf1;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #edf1f7, stop:0.6 #e5eaf1, stop:1 #dfe4ec);
    border: 1px solid #d3dae6;
    border-radius: 14px;
}
QWidget#histPanel { border-top: 1px solid #f4f7fb; border-left: 1px solid #f4f7fb; }
QWidget#playBar  { border-top: 1px solid #f4f7fb; border-left: 1px solid #f4f7fb; }

/* ===== 通用按钮：同底渐变凸起，按压缩进 ===== */
QPushButton {
    background-color: #e8ecf3;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f0f3f8, stop:0.55 #e6eaf1, stop:1 #dde3ec);
    color: #3a4356;
    border: 1px solid #c9d2e0;
    border-top-color: #eef1f6;
    border-left-color: #eef1f6;
    border-radius: 12px;
    padding: 6px 16px;
    min-height: 18px;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f4f6fb, stop:0.5 #eaeef5, stop:1 #e0e6ee);
    border: 1px solid #b9c6d8;
}
QPushButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #d9dfe9, stop:0.55 #e3e8f0, stop:1 #eaeef5);
    border: 1px solid #c2ccdb;
    color: #2f3850;
}
QPushButton:disabled {
    background-color: #eceff4;
    color: #a7afbf;
    border: 1px solid #dde3ec;
}
/* 彩色主按钮（id 选择器优先于上方通用规则，保持强调色） */
QPushButton#btnTranscribe, QPushButton#btnSummarize {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #5f8bf2, stop:1 #4271e6);
    color: #ffffff; border: 1px solid #4c7df0; border-radius: 12px; font-weight: bold;
}
QPushButton#btnTranscribe:hover, QPushButton#btnSummarize:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #6f97f4, stop:1 #527dea);
}
QPushButton#btnTranscribe:pressed, QPushButton#btnSummarize:pressed {
    background: #3a66d8;
}
QPushButton#btnRecord {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #ee6f6f, stop:1 #d94a4a);
    color: #ffffff; border: 1px solid #e05555; border-radius: 12px;
}
QPushButton#btnPlay {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #56b47e, stop:1 #3b9862);
    color: #ffffff; border: 1px solid #43a468; border-radius: 12px;
    min-width: 76px;
}
QPushButton#btnPlay:disabled { background-color:#eceff4; color:#a7afbf; border:1px solid #dde3ec; }
QPushButton#btnConfig, QPushButton#btnLoadModel { border-radius: 12px; }
QPushButton#btnLoadModel { border: 1px dashed #4c7df0; color: #3d6ddb; }

/* ===== 输入/文本区：内凹 ===== */
QLineEdit, QPlainTextEdit, QTextBrowser, QTextEdit {
    background-color: #dfe5ee;
    background: qlineargradient(x1:0, y1:1, x2:0, y2:0,
                stop:0 #dbe1eb, stop:1 #e6ebf2);
    color: #3a4356;
    border: 1px solid #cbd3e1;
    border-top: 1px solid #c3ccdb;
    border-left: 1px solid #c3ccdb;
    border-radius: 10px;
    padding: 6px;
    selection-background-color: #4c7df0;
    selection-color: #ffffff;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextBrowser:focus, QTextEdit:focus {
    border: 1px solid #4c7df0;
}
QComboBox {
    background-color: #e6eaf1;
    border: 1px solid #cbd3e1; border-radius: 10px;
    padding: 5px 8px; min-height: 18px; color: #3a4356;
}
QComboBox:hover { border-color: #4c7df0; }
QComboBox QAbstractItemView {
    background-color: #e8ecf3; color: #3a4356;
    border: 1px solid #cbd3e1; border-radius: 8px;
    selection-background-color: #4c7df0; selection-color: #ffffff;
    padding: 4px;
}

/* ===== 历史列表：透明底 + 圆角条目 ===== */
QListWidget {
    background: transparent;
    border: none;
    padding: 6px;
}
QListWidget::item {
    background-color: #e6eaf1;
    border: 1px solid #d5dce7;
    border-top-color: #eef1f6;
    border-left-color: #eef1f6;
    border-radius: 10px;
    padding: 7px 10px; margin: 3px 2px;
    color: #3a4356;
}
QListWidget::item:hover { background-color: #eef1f6; }
QListWidget::item:selected {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #5f8bf2, stop:1 #3d6ddb);
    color: #ffffff; border: 1px solid #3d6ddb;
}

/* ===== 播放进度滑块（内凹轨道；圆钮透明，保留可抓热区 2026-09-07） ===== */
QSlider::groove:horizontal {
    height: 8px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #cdd5e2, stop:1 #e2e7ee);
    border: 1px solid #cbd3e1; border-radius: 4px;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #7aa2f5, stop:1 #4c7df0);
    border-radius: 4px;
}
QSlider::add-page:horizontal { background: transparent; }
QSlider::handle:horizontal {
    width: 18px; margin: -6px 0;
    background: transparent;
    border: 1px solid transparent; border-radius: 9px;
}
QSlider::handle:horizontal:hover { border-color: transparent; }
QSlider::handle:horizontal:disabled { background: transparent; }

/* ===== 其余细节柔化 ===== */
QMenu { background-color: #e8ecf3; border: 1px solid #cdd5e2; border-radius: 10px; padding: 6px; }
QMenu::item { border-radius: 8px; color: #3a4356; }
QMenu::item:selected { background-color: #4c7df0; color: #ffffff; }
QMenu::separator { background: #cdd5e2; }
QToolTip { background-color: #e8ecf3; color: #3a4356; border: 1px solid #cdd5e2; border-radius: 6px; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background: #c2cbda; border-radius: 5px; }
QProgressBar { background-color: #dfe5ee; border: 1px solid #cdd5e2; border-radius: 8px; }
QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #6a92f5, stop:1 #4c7df0); border-radius: 7px; }
QSplitter::handle { background: #d3dae6; border-radius: 2px; }
QSplitter::handle:hover { background: #4c7df0; }
"""
