# 模型目录 / mod

本目录存放 FunASR 语音模型。**模型不入版本库**（约 5 GB），需按下表自行下载。

> 完整编译说明见仓库根目录的 [README.md](../README.md)。

## 目录名必须与下表完全一致

程序按固定名称查找模型，名字不对会报「模型缺失」。

| 本目录下的子目录 | 模型 | ModelScope 仓库 ID | 体积 | 必需性 |
|---|---|---|---|---|
| `fsmn-vad/` | fsmn-vad 语音活动检测 | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | 4 MB | **必需** |
| `cam++/` | cam++ 说话人嵌入 | `iic/speech_campplus_sv_zh-cn_16k-common` | 28 MB | **必需** |
| `sensevoice/` | SenseVoice-Small | `iic/SenseVoiceSmall` | 897 MB | 主模型三选一 |
| `paraformer/` | Paraformer-large | `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | 848 MB | 主模型三选一 |
| `fun-asr-nano/` | Fun-ASR-Nano-2512 | `FunAudioLLM/Fun-ASR-Nano-2512` | 2046 MB | 主模型三选一 |
| `ct-punc/` | ct-punc 标点恢复 | `iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch` | 1132 MB | **仅 Paraformer 需要** |

**主模型至少装一个**，程序默认使用 `sensevoice`。

## 下载命令

在**仓库根目录**执行（`mod/` 路径是相对的，别在别的目录跑）：

```bash
pip install modelscope

# 必需组件
modelscope download --model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch --local_dir mod/fsmn-vad
modelscope download --model iic/speech_campplus_sv_zh-cn_16k-common      --local_dir mod/cam++

# 主模型（按需选，至少一个）
modelscope download --model iic/SenseVoiceSmall                          --local_dir mod/sensevoice
modelscope download --model iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --local_dir mod/paraformer
modelscope download --model FunAudioLLM/Fun-ASR-Nano-2512                --local_dir mod/fun-asr-nano

# 仅 Paraformer 需要
modelscope download --model iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch --local_dir mod/ct-punc
```

**最小可用组合（CPU 机器，约 930 MB）**：
`fsmn-vad` + `cam++` + `sensevoice`

## 下载后自检

每个子目录内都应存在 `configuration.json` 和 `model.pt`：

```bat
dir mod\sensevoice
```

## 模型可以放在别处

若不想放这里（例如放在另一块盘共用），改 `config.json` 的 `model_dir` 为绝对路径：

```json
{ "model_dir": "D:\\models\\meeting" }
```
