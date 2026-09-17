# 智能会议纪要工具（Meeting）

> Windows 桌面应用：录音或导入音频 → 语音转文字（含**说话人分离**）→ AI 智能总结 → 导出会议纪要。
> 识别与说话人分离**全部在本机完成**，音频不上传；仅「AI 总结」一项需要 DeepSeek API Key。

技术栈：Python 3.11 + PyQt6 + FunASR（进程内推理）+ PyTorch + ffmpeg。

---

## ⚠️ 先读这一段：clone 下来为什么直接跑不起来

模型与 ffmpeg 合计约 **5 GB**，体积大且可重新下载，因此**不纳入版本库**
（`.gitignore` 已排除 `runtime/`）。

这意味着你需要自己做两件事：

| 缺什么 | 去哪拿 | 放到哪 |
|---|---|---|
| 语音模型（6 个目录，约 5 GB） | ModelScope（见 [第四节](#四模型下载与放置)） | `<仓库根>/mod/` |
| `ffmpeg.exe`（约 103 MB） | gyan.dev（见 [第五节](#五ffmpeg-下载与放置)） | `<仓库根>/bin/` |

代码本身是完整的，补齐这两项即可编译运行。

---

## 一、功能概览

| 功能 | 说明 |
|---|---|
| 录音 / 导入 | 麦克风录制，或导入 mp3 / wav / m4a / flac |
| 语音转写 | 三种模型可选（见下表），中文会议向优化 |
| 说话人分离 | 自动区分「谁在说话」，句级时间戳 + 说话人标签 |
| 说话人编辑 | 重命名、合并说话人 |
| 播放联动 | 播放时按句高亮，点时间戳跳转，空格键播放/暂停 |
| AI 总结 | DeepSeek，内置 4 套模板（会议纪要等） |
| 记录管理 | 每条记录独立目录（音频 + 转写 + 摘要 + 元数据） |
| 主题 | 浅色 / 深色，即时切换 |

**三种识别模型**（`config.json` 的 `asr_model` 切换）：

| 值 | 模型 | 特点 |
|---|---|---|
| `sensevoice` | SenseVoice-Small | **默认**。约 0.9 GB，CPU 可实时，50+ 语种，自带标点 |
| `paraformer` | Paraformer-large | 约 0.85 GB，纯中文精度最好，输出**字级**时间戳 |
| `fun-asr-nano` | Fun-ASR-Nano-2512 | 约 2 GB，LLM 架构，质量最高，**建议配 GPU** |

---

## 二、环境要求

| 项 | 要求 |
|---|---|
| 操作系统 | Windows 10 / 11，**64 位** |
| Python | **3.11**（已在 3.11.10 验证；3.12+ 未验证，不建议） |
| GPU | 可选。NVIDIA 显卡 + 驱动 **≥ 530.41** 可启用 CUDA 12.1 加速；无 GPU 自动降级 CPU |
| 磁盘 | 建议预留 **15 GB** 以上（模型约 5 GB + PyTorch 约 2.5 GB + 依赖与构建产物） |
| 内存 | 建议 16 GB 以上（Paraformer / Fun-ASR-Nano 加载时占用较高） |

**性能参考**（开发机实测，无 GPU 的 CPU 推理，4 分 45 秒录音）：
模型加载约 60–84 秒，转写约 135 秒。首次加载后模型常驻内存，后续转写无需重复加载。

---

## 三、从零编译（六步）

### 第 1 步：取代码

```bash
git clone https://github.com/liliZhl/Meeting.git
cd Meeting
```

### 第 2 步：建虚拟环境

```bat
py -3.11 -m venv venv
venv\Scripts\activate
```

### 第 3 步：安装 PyTorch

**有 NVIDIA 显卡（CUDA 12.1）：**

```bat
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**无显卡 / 只要 CPU：**

```bat
pip install torch==2.5.1 torchaudio==2.5.1
```

> 装完可自检：`python -c "import torch; print(torch.cuda.is_available())"`，
> 输出 `True` 表示 GPU 可用。

### 第 4 步：安装 FunASR（**必须从源码装**）

```bat
pip install git+https://github.com/modelscope/FunASR.git
```

> **为什么不能用 `pip install funasr`**：说话人分离（cam++）依赖源码仓库中的能力，
> 发布到 PyPI 的包不包含这部分。用发布版会在转写时缺少说话人结果。

### 第 5 步：安装其余依赖

```bat
pip install -r requirements.txt
```

> 若国内下载慢，可加镜像：
> `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 第 6 步：下载模型与 ffmpeg

分别见 [第四节](#四模型下载与放置) 与 [第五节](#五ffmpeg-下载与放置)。
完成后即可运行：`venv\Scripts\python.exe -X utf8 src\main.py`

---

## 四、模型下载与放置

### 4.1 放在哪

```
Meeting/                      ← 仓库根目录
├── mod/                      ← ★ 模型都放这里
│   ├── fsmn-vad/
│   ├── cam++/
│   ├── sensevoice/
│   ├── paraformer/
│   ├── fun-asr-nano/
│   └── ct-punc/
├── bin/
│   └── ffmpeg.exe            ← ★ ffmpeg 放这里
└── src/
```

**目录名必须与上表完全一致** —— 程序按固定名称查找模型，名字不对会报「模型缺失」。

### 4.2 要下载哪些

| 放置目录 | 模型 | ModelScope 仓库 ID | 体积 | 必需性 |
|---|---|---|---|---|
| `mod/fsmn-vad/` | fsmn-vad 语音活动检测 | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | 4 MB | **必需** |
| `mod/cam++/` | cam++ 说话人嵌入 | `iic/speech_campplus_sv_zh-cn_16k-common` | 28 MB | **必需** |
| `mod/sensevoice/` | SenseVoice-Small | `iic/SenseVoiceSmall` | 897 MB | 主模型三选一 |
| `mod/paraformer/` | Paraformer-large | `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | 848 MB | 主模型三选一 |
| `mod/fun-asr-nano/` | Fun-ASR-Nano-2512 | `FunAudioLLM/Fun-ASR-Nano-2512` | 2046 MB | 主模型三选一 |
| `mod/ct-punc/` | ct-punc 标点恢复 | `iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch` | 1132 MB | **仅 Paraformer 需要** |

**主模型至少装一个**，程序默认使用 `sensevoice`。

**按机器选组合：**

| 场景 | 需要下载 | 合计 |
|---|---|---|
| CPU 机器，最小可用 | `sensevoice` + `fsmn-vad` + `cam++` | 约 930 MB |
| 中文精度优先 | `paraformer` + `ct-punc` + `fsmn-vad` + `cam++` | 约 2.0 GB |
| 全量（可在三种模型间自由切换） | 上表六项全下 | 约 4.9 GB |

> `ct-punc` 只有 Paraformer 需要 —— SenseVoice 与 Fun-ASR-Nano 自带标点。

### 4.3 下载方法

#### 方法 A：命令行（推荐）

```bash
pip install modelscope
```

然后逐条执行（`--local_dir` 直接指向 `mod/` 下的目标目录）：

```bash
# 必需组件
modelscope download --model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch --local_dir mod/fsmn-vad
modelscope download --model iic/speech_campplus_sv_zh-cn_16k-common      --local_dir mod/cam++

# 主模型（按需选）
modelscope download --model iic/SenseVoiceSmall                          --local_dir mod/sensevoice
modelscope download --model iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --local_dir mod/paraformer
modelscope download --model FunAudioLLM/Fun-ASR-Nano-2512                --local_dir mod/fun-asr-nano

# 仅 Paraformer 需要
modelscope download --model iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch --local_dir mod/ct-punc
```

ModelScope 服务器在国内，直连即可，**不需要代理**。

#### 方法 B：网页手动下载

1. 打开 `https://www.modelscope.cn/models/<上表的仓库 ID>`
2. 进入「模型文件」标签页
3. 逐个下载全部文件，放进上表对应的目录

> **注意**：网页下载的压缩包解压后，目录名常带 `--` 前缀
> （例如 `iic--SenseVoiceSmall`），**必须重命名**为上表左列的短名
> （`sensevoice`），否则程序找不到模型。

#### 方法 C：从一台已经装好的机器整目录拷贝

模型文件与机器无关，直接复制 `mod/` 整个目录最省事：

```bat
robocopy D:\Meeting\mod \\目标机\share\Meeting\mod /E
```

U 盘拷贝同理。目录名保持一致即可，无需任何额外配置。

### 4.4 验证是否下载到位

```bat
dir mod
```

应能看到你选择的那几个子目录；每个子目录内都应含 `configuration.json` 与 `model.pt`
（`fsmn-vad` 的 `model.pt` 只有 1.6 MB，属正常）。

### 4.5 如果程序仍然提示模型缺失

`config.json` 里的 `model_dir` 默认是相对路径 `mod`，由程序自动解析为仓库根下的 `mod/`。
若你的模型放在别处（例如另一个盘的目录），把它改成**绝对路径**即可：

```json
{
  "model_dir": "D:\\models\\meeting"
}
```

---

## 五、ffmpeg 下载与放置

**用途**：把导入的 mp3 / m4a 等格式统一转成 16 kHz 单声道 wav 再送识别。
没有 ffmpeg 时部分格式会加载失败（尤其 m4a）。

**来源**：gyan.dev 维护的 FFmpeg Windows 官方风格构建（64 位静态版）

| 项 | 值 |
|---|---|
| 下载页 | https://www.gyan.dev/ffmpeg/builds/ |
| 直链（release essentials，zip） | https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip |
| 体积 | 约 106 MB |
| 本项目验证版本 | **ffmpeg 9.0.1**（解压后目录名 `ffmpeg-9.0.1-essentials_build`） |

**步骤：**

1. 下载 `ffmpeg-release-essentials.zip` 并解压
2. 进入 `ffmpeg-9.0.1-essentials_build/bin/`
3. 取出其中的 **`ffmpeg.exe`**
4. 放到 `<仓库根>/bin/ffmpeg.exe`（`bin/` 目录不存在就新建）

只需 `ffmpeg.exe` 一个文件，`ffplay.exe` / `ffprobe.exe` 不需要。

> 兜底：如果不想下载，只要已把 ffmpeg 装进系统 PATH（例如 `winget install ffmpeg`），
> 程序也能找到。但**自带 `bin/ffmpeg.exe` 更可靠**，打包分发时也依赖它。

---

## 六、配置说明

首次运行会在应用目录生成 `config.json`（该文件已被 `.gitignore` 排除，不会提交）：

| 键 | 默认值 | 说明 |
|---|---|---|
| `deepseek_api_key` | 空 | AI 总结所需的 DeepSeek Key，从 https://platform.deepseek.com 获取 |
| `deepseek_base_url` | `https://api.deepseek.com` | 接口地址，兼容 OpenAI 协议的服务可替换 |
| `deepseek_model` | `deepseek-chat` | 总结所用模型 |
| `theme` | `light` | 界面主题 `light` / `dark` |
| `summary_template` | `meeting` | 总结模板 |
| `model_dir` | `mod` | 模型目录，相对路径或绝对路径 |
| `asr_model` | `sensevoice` | 识别模型：`sensevoice` / `fun-asr-nano` / `paraformer` |
| `vad_level` | `medium` | 切句灵敏度：`fine`(0.8s) / `medium`(1.5s) / `coarse`(2.5s) |

**`vad_level` 怎么选**（实测参考，4 分 45 秒录音）：
`fine` → 105 段（过碎）；`medium` → 29 段（推荐，会议平衡）；`coarse` → 16 段（长句优先，
但可能把不同说话人合并）。

首次运行时程序会弹出配置引导，指引填写 API Key。不配置 Key 时其余功能照常可用，
仅「AI 总结」不可用。

---

## 七、运行（开发模式）

```bat
venv\Scripts\python.exe -X utf8 src\main.py
```

> `-X utf8` 用于避免 Windows 控制台编码问题。建议始终带上。

语法自检：

```bat
venv\Scripts\python.exe -m py_compile src\*.py
```

运行日志写在 `logs/app_YYYYMMDD.log`，排查问题先看它。

---

## 八、打包 EXE

```bat
venv\Scripts\python.exe -m PyInstaller build_exe.spec --log-level INFO
```

产物为 `dist/MeetingAssistant/`（onedir 模式）。

**打包后必须把模型与 ffmpeg 拷进产物目录**（`build_exe.spec` 刻意不打包它们）：

```bat
xcopy mod dist\MeetingAssistant\mod /E /I
xcopy bin dist\MeetingAssistant\bin /E /I
```

最终分发时整个 `dist\MeetingAssistant\` 文件夹拷给对方即可，目标机**无需安装 Python**。

**几个已知注意点：**

1. **打包前先关闭正在运行的 EXE**，否则 `dist/` 内文件被占用，构建会失败。
2. **旧 `dist/` 目录建议先改名隔离再打包**（例如加 `_bak` 后缀）。
   直接覆盖时 PyInstaller 的 COLLECT 阶段可能静默失败，产物不完整。
3. 完整构建约需 **8 分钟**，中途不要用短超时打断。
4. 打包过程中若被强杀，`%LOCALAPPDATA%\torch_extensions\` 下可能残留 0 字节锁文件，
   导致下次构建**无限挂起**。遇到这种情况先清掉该缓存目录再重试。

---

## 九、目录结构

```
Meeting/
├── src/                          # 源码
│   ├── main.py                   # 主程序：窗口、导航、录音、转写流程、AI 总结
│   ├── asr_engine.py             # ASR 引擎：模型加载、转写、说话人分离、结果结构化
│   ├── model_manager.py          # 模型单例管理：常驻内存 + 启动后台预加载
│   ├── player.py                 # 音频播放封装
│   ├── store.py                  # 记录存储（每条记录一个目录 + 索引）
│   └── theme.py                  # 主题与 QSS
├── mod/                          # ★ 模型目录（不入库，需自行下载）
├── bin/                          # ★ ffmpeg.exe（不入库，需自行下载）
├── assets/                       # 图标等静态资源
├── docs/                         # 设计与技术文档
├── memory/                       # 开发过程记录
├── build_exe.spec                # PyInstaller 打包配置
└── requirements.txt              # 依赖清单
```

运行时还会生成（均已忽略，不入库）：
`logs/`（日志）、`records/`（转写记录）、`config.json`（配置）、`build/` `dist/`（构建产物）。

---

## 十、常见问题

**Q：启动报「模型缺失：sensevoice」**
模型没下或目录名不对。检查 `mod/sensevoice/` 是否存在、里面有没有 `configuration.json`；
若模型在别处，把 `config.json` 的 `model_dir` 改为绝对路径。

**Q：选 Paraformer 后报缺 `ct-punc`**
Paraformer 不带标点，必须额外下载 `ct-punc`（见 4.2）。

**Q：导入 m4a / mp3 转写失败或卡住**
缺 `bin/ffmpeg.exe`。按第五节补上，这是最常见的失败原因。

**Q：识别结果里出现 `幺幺幺…`、`包子包子…` 之类的重复**
这是 LLM 架构模型（Fun-ASR-Nano）的解码重复现象。代码已内置抑制参数
（`repetition_penalty` / `no_repeat_ngram_size`），若仍明显，改用 `sensevoice` 或
`paraformer`（非 LLM 架构，无此问题）。

**Q：没有 GPU 能用吗？**
能。会自动降级 CPU（日志里能看到「未检测到 GPU，使用 CPU」）。
CPU 下建议用 `sensevoice`，可接近实时；`fun-asr-nano` 在 CPU 上约为 0.5 倍实时，偏慢。

**Q：装了 GPU 版 PyTorch 但走的是 CPU**
检查显卡驱动版本是否 ≥ 530.41（CUDA 12.1 要求），以及是否误装了 CPU 版 torch。
用 `python -c "import torch; print(torch.cuda.is_available())"` 验证。

**Q：模型加载很慢（一分钟以上）**
属正常现象，尤其首次加载。模块设计上模型加载一次后常驻内存，且应用启动后会后台预加载。

---

## 十一、相关文档

| 文档 | 内容 |
|---|---|
| [docs/技术文档_20260904.md](docs/技术文档_20260904.md) | 完整技术文档：架构、模块职责、打包细节、踩坑备忘录 |
| [docs/说话人分离方案_20260907.md](docs/说话人分离方案_20260907.md) | 说话人分离的技术选型与实测结论 |
| [docs/幻觉抑制_20260902.md](docs/幻觉抑制_20260902.md) | LLM 架构模型的重复输出抑制 |
| [docs/使用说明.md](docs/使用说明.md) | 使用说明与操作指引 |
| [docs/规划.md](docs/规划.md) | 原始规划与实施进度 |

---

## 十二、许可证

本项目源码采用 [MIT 许可证](LICENSE)：可自由使用、修改、分发与商用，只需保留版权声明。

### 第三方组件与素材

**MIT 只覆盖本仓库的源码**，不改变下列依赖各自的许可要求。商用或闭源分发前请自行核对：

| 组件 | 许可 / 来源 | 备注 |
|---|---|---|
| PyQt6 | GPL v3 或商业双许可（Riverbank） | ⚠️ 见下方提示 |
| FunASR | MIT（阿里达摩院） | 可自由使用 |
| 语音模型（Fun-ASR-Nano / SenseVoice / Paraformer / cam++ / fsmn-vad / ct-punc） | 见 ModelScope 各模型页的 License 字段 | 各模型许可不同，逐一核对 |
| ffmpeg | LGPL 2.1+ 或 GPL（取决于构建选项） | 本仓库不含二进制，由使用者自行下载 |
| DeepSeek API | 服务条款（非开源许可） | 需自行申请 Key |
| `assets/` 中的应用图标 | 外部素材，来源与授权待确认 | 商用前请替换为自有素材或核实授权 |

> ⚠️ **PyQt6 是 GPL v3 / 商业双许可**。若你计划以**闭源**方式分发本工具的衍生版本，
> 需要自行取得 Riverbank 的商业许可；只按 MIT 分发源码并不能豁免这一要求。
> 想完全避开该问题，可改用 LGPL 的 PySide6（二者 API 高度兼容）。

> 本项目仅用于个人学习与自用；使用者需自行确认其使用场景符合所在组织的相关规定。

