# Live-Stream Whisper

<video src="ASR Demo.mp4" width="800" controls></video>

基于 [Simul-Whisper](https://arxiv.org/pdf/2406.10052)（INTERSPEECH 2024 论文：Attention-Guided Streaming Whisper）的实时音频转录服务器。

音频通过 WebSocket 传输，经 Silero VAD + Simul-Whisper 处理后实时返回转录结果。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载 Whisper 模型

| 模型 | 下载链接 |
|------|----------|
| base | [base.pt](https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt) |
| small | [small.pt](https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt) |
| medium | [medium.pt](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt) |
| large-v2 | [large-v2.pt](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt) |

将模型文件放入 `./models/whisper/`（或通过 `WHISPER_MODEL_PATH` 环境变量指定路径）。

### 3. 启动服务器

```bash
python start_server.py
```

服务器监听 `ws://localhost:8765`。

## 配置

通过环境变量配置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WHISPER_MODEL_PATH` | `./models/whisper/base.pt` | Whisper 模型路径 |
| `WHISPER_LANGUAGE` | `en` | 语言代码（`en`、`zh`、`auto` 等） |
| `WHISPER_SEGMENT_LENGTH` | `0.064` | 音频分块时长（秒） |
| `WHISPER_FRAME_THRESHOLD` | `4` | 注意力引导解码阈值 |
| `WHISPER_AUDIO_MAX_LEN` | `5.0` | 最大音频缓冲时长（秒） |
| `SERVER_PORT` | `8765` | WebSocket 服务器端口 |

## WebSocket 协议

**客户端 → 服务器**：发送原始 16 位 PCM 音频字节（16000Hz 单声道，每条消息一个音频块）。

**服务器 → 客户端**：
```json
{"type": "ready", "session_id": "...", "sample_rate": 16000}
{"type": "partial", "text": "转录文本"}
{"type": "done"}
{"type": "error", "message": "..."}
```

**命令**（文本消息）：`stop`、`eof`、`flush` 结束音频流；`ping` 返回 `{"type": "pong"}`。

## Python 客户端示例

```python
import asyncio, websockets, json, numpy as np

async def main():
    async with websockets.connect("ws://localhost:8765") as ws:
        # 接收连接就绪消息
        print(await ws.recv())

        # 发送音频（16位 PCM，16000Hz 单声道）
        audio = (np.sin(2 * np.pi * 440 * np.arange(0, 1, 1/16000)) * 32767).astype(np.int16)
        await ws.send(audio.tobytes())

        # 接收转录结果
        print(await ws.recv())

asyncio.run(main())
```

## 系统架构

```
server/websocket_server.py     # WebSocket 处理，音频队列
    └── core/transcriber.py     # 处理流水线：VAD 静音跳过 → Simul-Whisper
            ├── engines/vad/   # Silero VAD（丢弃静音块）
            └── engines/simul_whisper/  # 流式 Whisper
                    └── transcriber/simul_whisper.py  # PaddedAlignAttWhisper
                    └── whisper/                # OpenAI Whisper 编码器
```

## 修复说明

本库修复了原 [simul_whisper](live_stream_whisper/whisper/simul_whisper/) 中 Triton 中值滤波操作的兼容性问题：

- 原代码使用 `JITFunction.src` 源码动态替换的方式仅适用于 Triton 2.x，在 Triton 3.x 下会因 API 变化导致运行错误
- 本库新增 Triton 3.x 兼容路径，对 filter_width 为 3/5/7 的情况使用 CUDA sort 实现中值滤波，其余回退到标准排序路径

同时，通过调整 `WHISPER_SEGMENT_LENGTH`，可将端到端延迟控制在 400ms–600ms 范围内。

## 测试

离线转录测试（需要 `tests/english.wav` 或修改 `test_client.py` 中的 `audio_path`）：

```bash
python test_client.py
```

## 参考

- [Simul-Whisper](https://arxiv.org/pdf/2406.10052) — INTERSPEECH 2024
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad)
