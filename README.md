# Live-Stream Whisper

<video src="https://github.com/user-attachments/assets/b50ff7ab-6236-4a76-9f14-88c48e989eda" width="800" controls></video>

Real-time audio transcription server using [Simul-Whisper](https://arxiv.org/pdf/2406.10052) — attention-guided streaming Whisper from INTERSPEECH 2024.

Audio is streamed via WebSocket, processed through Silero VAD + Simul-Whisper, and partial transcription results are returned in real time.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Whisper Model

| Model | Download |
|-------|----------|
| base | [base.pt](https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt) |
| small | [small.pt](https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt) |
| medium | [medium.pt](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt) |
| large-v2 | [large-v2.pt](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt) |

Place the model in `./models/whisper/` (or set `WHISPER_MODEL_PATH`).

### 3. Start Server

```bash
python start_server.py
```

Server listens on `ws://localhost:8765`.

## Configuration

Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL_PATH` | `./models/whisper/base.pt` | Whisper model path |
| `WHISPER_LANGUAGE` | `en` | Language code (`en`, `zh`, `auto`, etc.) |
| `WHISPER_SEGMENT_LENGTH` | `0.064` | Audio chunk size in seconds |
| `WHISPER_FRAME_THRESHOLD` | `4` | Attention-guided decoding threshold |
| `WHISPER_AUDIO_MAX_LEN` | `5.0` | Max audio buffer in seconds |
| `SERVER_PORT` | `8765` | WebSocket server port |

## WebSocket Protocol

**Client → Server**: Send raw 16-bit PCM audio bytes (16000Hz mono, single chunk per message).

**Server → Client**:
```json
{"type": "ready", "session_id": "...", "sample_rate": 16000}
{"type": "partial", "text": "transcribed text"}
{"type": "done"}
{"type": "error", "message": "..."}
```

**Commands** (text messages): `stop`, `eof`, `flush` end the audio stream; `ping` returns `{"type": "pong"}`.

## Example Client (Python)

```python
import asyncio, websockets, json, numpy as np

async def main():
    async with websockets.connect("ws://localhost:8765") as ws:
        # Receive ready
        print(await ws.recv())

        # Send audio (16-bit PCM, 16000Hz mono)
        audio = (np.sin(2 * np.pi * 440 * np.arange(0, 1, 1/16000)) * 32767).astype(np.int16)
        await ws.send(audio.tobytes())

        # Receive transcription
        print(await ws.recv())

asyncio.run(main())
```

## Architecture

```
server/websocket_server.py     # WebSocket handler, audio queue
    └── core/transcriber.py     # Pipeline: VAD silence skip → Simul-Whisper
            ├── engines/vad/   # Silero VAD (drops silent chunks)
            └── engines/simul_whisper/  # Streaming Whisper
                    └── transcriber/simul_whisper.py  # PaddedAlignAttWhisper
                    └── whisper/                # OpenAI Whisper encoder
```

## Fixes

This library fixes Triton median filter compatibility issues from the original [simul_whisper](live_stream_whisper/whisper/simul_whisper/):

- The original code uses `JITFunction.src` source patching, which only works on Triton 2.x and breaks on Triton 3.x due to API changes
- This library adds a Triton 3.x compatible path using CUDA sort for filter_width of 3/5/7, with a fallback to standard sort for other sizes

Additionally, by tuning `WHISPER_SEGMENT_LENGTH`, end-to-end latency can be controlled within 400ms–600ms.

## Testing

Offline transcription test (requires a test audio file at `tests/english.wav` or modify `audio_path` in `test_client.py`):

```bash
python test_client.py
```

## Reference

- [Simul-Whisper](https://arxiv.org/pdf/2406.10052) — INTERSPEECH 2024
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad)
