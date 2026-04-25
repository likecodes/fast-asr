# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Real-time audio transcription server using Simul-Whisper (attention-guided streaming Whisper from INTERSPEECH 2024). Audio is received via WebSocket, processed through VAD + streaming Whisper, and partial/transcribed text is returned via WebSocket.

## Running the Server

```bash
python start_server.py
```

WebSocket server listens on `ws://localhost:8765` by default.

## Configuration

All settings are environment variables with defaults in `config/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL_PATH` | `./models/whisper/base.pt` | Path to Whisper model |
| `WHISPER_LANGUAGE` | `en` | Language code or `auto` |
| `WHISPER_SEGMENT_LENGTH` | `0.064` | Chunk size in seconds |
| `WHISPER_FRAME_THRESHOLD` | `4` | Attention-guided decoding threshold |
| `WHISPER_AUDIO_MAX_LEN` | `5.0` | Max audio buffer in seconds |
| `SERVER_PORT` | `8765` | WebSocket server port |

## WebSocket Protocol

**Client → Server**: Send raw 16-bit PCM audio bytes (16000Hz mono).

**Server → Client**:
- `{"type": "ready", "session_id": "...", "sample_rate": 16000}` — connection established
- `{"type": "partial", "text": "..."}` — partial transcription result
- `{"type": "done"}` — transcription complete
- `{"type": "error", "message": "..."}` — error occurred

**Client → Server commands** (text messages):
- `stop` / `eof` / `flush` — end audio stream
- `ping` — server responds with `{"type": "pong"}`

## Architecture

```
server/websocket_server.py     # WebSocket connection handler
    └── core/transcriber.py     # Audio processing pipeline
            ├── engines/vad/   # Silero VAD (skips silent chunks)
            └── engines/simul_whisper/  # Simul-Whisper model
                    └── whisper/         # OpenAI Whisper encoder
                    └── transcriber/     # PaddedAlignAttWhisper (attention-guided streaming)
```

**Audio flow**: WebSocket → `audio_queue` → VAD silence check → `infer_segment()` → partial result via WebSocket

**Key files**:
- `engines/simul_whisper/transcriber/simul_whisper.py` — `PaddedAlignAttWhisper` class, implements attention-guided streaming decoding with CIF-based boundary detection
- `engines/vad/vad.py` — `VAD` class wrapping Silero VAD with chunk-based silence detection
- `core/transcriber.py` — `Transcriber` class, orchestrates VAD + Whisper inference per audio chunk

## Test Client

`test_client.py` contains both:
1. WebSocket client — connects to server and streams audio
2. Direct model test — loads `PaddedAlignAttWhisper` directly for offline testing

To test offline transcription:
```bash
python test_client.py
```
(Requires `tests/english.wav` or modify `audio_path`)

## Dependencies

Core: `torch`, `websockets`, `numpy`
VAD: `torchaudio`, `silero-vad`
Whisper: `faster-whisper` (optional encoder acceleration), `tiktoken`, `regex`, `numba`