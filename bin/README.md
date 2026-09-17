# 外部工具目录 / bin

本目录存放 ffmpeg 可执行文件。**不入版本库**（单个文件约 103 MB），需自行下载。

> 完整编译说明见仓库根目录的 [README.md](../README.md)。

## 需要放什么

只需要一个文件：

```
bin/
└── ffmpeg.exe        ← 就这一个，约 103 MB
```

`ffplay.exe` / `ffprobe.exe` **不需要**。

## 从哪下载

| 项 | 值 |
|---|---|
| 来源 | gyan.dev 的 FFmpeg Windows 构建（64 位静态版） |
| 下载页 | https://www.gyan.dev/ffmpeg/builds/ |
| 直链 | https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip |
| 体积 | 约 106 MB |
| 本项目验证版本 | **ffmpeg 9.0.1** |

## 步骤

1. 下载 `ffmpeg-release-essentials.zip`
2. 解压，进入 `ffmpeg-9.0.1-essentials_build/bin/`
3. 把其中的 **`ffmpeg.exe`** 复制到本目录
4. 确认最终路径为 `<仓库根>/bin/ffmpeg.exe`

## 为什么需要它

导入的 mp3 / m4a / flac 等格式需要先统一转成 16 kHz 单声道 wav 再送识别。
缺了它，m4a 这类格式会加载失败或卡住。

> 兜底方案：若系统 PATH 中已有 ffmpeg（例如 `winget install ffmpeg`），
> 程序也能找到。但自带 `bin/ffmpeg.exe` 更可靠，打包分发时也依赖它。
