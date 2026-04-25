"""
WebSocket 实时转写服务器

职责：
- 接收前端通过 WebSocket 发送的音频（二进制 PCM16/mono/16k）
- 将音频放入 per-session 队列，并由消费者按照 segment_length 组片段
- 通过 simul_whisper(PaddedAlignAttWhisper) 增量转写
- 将增量转写结果放入结果队列，并推送回前端
"""

import asyncio
import logging
import uuid
import json
from typing import Dict, Optional

import torch
import time
import os

# 本地模块
from config.config import config_manager
from utils.tools import audio_to_tensor
from core.types import SAMPLING_RATE

# WebSocket
import websockets
from websockets.server import WebSocketServerProtocol

# simul_whisper 对齐注意力增量转写
from engines.simul_whisper.transcriber import AlignAttConfig, PaddedAlignAttWhisper


logger = logging.getLogger(__name__)


class WebSocketTranscriptionServer:
    def __init__(self, host: str = "localhost", port: int = 8765):
        cfg = config_manager.get_config()
        self._server_cfg = cfg["server"]
        self._whisper_cfg = cfg["whisper"]

        # 允许覆盖传入 host/port
        self.host = host or self._server_cfg.host
        self.port = port or self._server_cfg.port

        # per-connection 状态
        self._connections: Dict[str, Dict[str, object]] = {}
        self._server = None
        self._stop_event = asyncio.Event()

        # 初始化推理配置（实例化在每个会话内完成，避免共享状态冲突）
        self._align_cfg = AlignAttConfig(
            model_path=self._whisper_cfg.model_path,
            segment_length=float(self._whisper_cfg.segment_length),
            min_seg_len=float(self._whisper_cfg.min_seg_len),
            language=self._whisper_cfg.language,
            frame_threshold=int(self._whisper_cfg.frame_threshold),
            nonspeech_prob=float(self._whisper_cfg.nonspeech_prob),
            rewind_threshold=int(self._whisper_cfg.rewind_threshold),
            # 使用默认 buffer/min_seg；如需更强控制，可在配置中新增并映射

            if_ckpt_path=self._whisper_cfg.cif_ckpt_path or "",
        )

        # 片段长度（样本）
        self._segment_samples = int(self._whisper_cfg.segment_length * SAMPLING_RATE)
        # VAD 阈值（RMS），默认 0.005，可通过环境变量 VAD_RMS_THRESHOLD 调整
        try:
            self._vad_rms_threshold = float(os.getenv("VAD_RMS_THRESHOLD", "0.005"))
        except Exception:
            self._vad_rms_threshold = 0.005

    async def start_server(self):
        logger.info(f"启动 WebSocket 服务器 ws://{self.host}:{self.port}")
        async with websockets.serve(
            self._handler,
            self.host,
            self.port,
            max_size=self._server_cfg.max_size,
            ping_interval=self._server_cfg.ping_interval,
            ping_timeout=self._server_cfg.ping_timeout,
        ):
            # 等待 stop 信号
            await self._stop_event.wait()

    async def stop(self):
        try:
            self._stop_event.set()
        except Exception:
            pass

    async def _handler(self, websocket: WebSocketServerProtocol):
        session_id = str(uuid.uuid4())
        audio_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

        # 累积 buffer
        buffer_chunks = []  # list[torch.Tensor]
        buffer_total = 0

        # 记录连接状态
        self._connections[session_id] = {
            "websocket": websocket,
            "audio_queue": audio_queue,
            "result_queue": None,
        }

        logger.info(f"新连接: {session_id}")

        # 不立即发送 ready，等待所有模型加载并预热完成后再通知

        producer_task = asyncio.create_task(
            self._receive_audio_loop(session_id, websocket, audio_queue)
        )
        # 为本会话创建独立的转写器
        transcriber = PaddedAlignAttWhisper(self._align_cfg)
        # 加载 silero VAD 模型（每会话一份，避免共享状态冲突）
        silero_model = None
        vad_threshold = 0.5
        try:
            silero_model, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
            vad_cfg = config_manager.get_config().get("vad", None)
            if vad_cfg is not None and hasattr(vad_cfg, 'threshold'):
                vad_threshold = float(_vadcfg.threshold)
            logger.info(f"Silero VAD 模型加载成功, threshold={vad_threshold}")
        except Exception as e:
            logger.warning(f"Silero VAD 加载失败: {e}")
        # 预热：跑一小段静音以初始化缓存和编译路径
        try:
            warmup_seg = torch.zeros(int(0.1 * SAMPLING_RATE))
            _ = transcriber.infer(warmup_seg, True)
        except Exception as e:
            logger.debug(f"warmup 失败: {e}")

        # 模型+VAD就绪后再告知前端可以推流
        try:
            await websocket.send(
                json.dumps(
                    {
                        "type": "ready",
                        "session_id": session_id,
                        "segment_length": float(self._whisper_cfg.segment_length),
                        "sample_rate": SAMPLING_RATE,
                    },
                    ensure_ascii=False,
                )
            )
        except Exception:
            pass

        # 启动协程消费者与结果推送
        consumer_task = asyncio.create_task(
            self._transcribe_loop(session_id, websocket, audio_queue, buffer_chunks, transcriber, silero_model, vad_threshold)
        )

        try:
            done, pending = await asyncio.wait(
                {producer_task, consumer_task},
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for t in done:
                exc = t.exception()
                if exc:
                    raise exc
        except Exception as e:
            logger.error(f"会话 {session_id} 发生错误: {e}")
        finally:
            # 结束所有任务
            try:
                await audio_queue.put(None)
            except Exception:
                pass
            for task in (producer_task, consumer_task):
                if not task.done():
                    task.cancel()
            # 移除连接
            self._connections.pop(session_id, None)
            logger.info(f"连接关闭: {session_id}")

    async def _receive_audio_loop(
        self,
        session_id: str,
        websocket: WebSocketServerProtocol,
        audio_queue: asyncio.Queue,
    ):
        try:
            async for message in websocket:
                # 二进制为裸 PCM 帧；文本为 JSON 指令
                if isinstance(message, (bytes, bytearray)):
                    try:
                        await audio_queue.put(bytes(message))
                    except Exception:
                        logger.warning(f"音频队列写入失败，丢弃一帧: {session_id}")
                elif isinstance(message, str):
                    # 文本作为控制指令：stop/eof/flush/ping；不再接受 base64 音频
                    cmd = message.strip()
                    if cmd.startswith('{') and cmd.endswith('}'):
                        try:
                            data = json.loads(cmd)
                            cmd = str(data.get("type", "")).lower()
                        except Exception:
                            cmd = cmd.lower()
                    else:
                        cmd = cmd.lower()

                    if cmd in {"stop", "eof", "flush"}:
                        try:
                            await audio_queue.put(None)
                        except Exception:
                            pass
                    elif cmd == "ping":
                        try:
                            await websocket.send(json.dumps({"type": "pong"}))
                        except Exception:
                            pass
                    # 其他文本忽略
                # 其他类型保留
        except Exception as e:
            logger.warning(f"接收端异常 {session_id}: {e}")
        finally:
            # 确保退出时通知消费者
            try:
                await audio_queue.put(None)
            except Exception:
                pass

    async def _transcribe_loop(
        self,
        session_id: str,
        websocket: WebSocketServerProtocol,
        audio_queue: asyncio.Queue,
        buffer_chunks: list,
        transcriber: PaddedAlignAttWhisper,
        silero_model,
        vad_threshold: float,
    ):
        buffer_total = 0

        def is_silent(audio_tensor: torch.Tensor) -> bool:
            if audio_tensor is None or audio_tensor.numel() == 0:
                return True
            # 优先用 silero VAD 判断
            if silero_model is not None:
                try:
                    x = audio_tensor
                    if x.dim() == 1:
                        x = x.unsqueeze(0)
                    prob = float(silero_model(x, SAMPLING_RATE).item())
                    return prob < vad_threshold
                except Exception as e:
                    logger.debug(f"silero_vad 判定失败，回退 RMS: {e}")
            # 回退：RMS 门限
            rms = torch.sqrt(torch.mean(audio_tensor.float() * audio_tensor.float()))
            try:
                rms_val = float(rms.item())
            except Exception:
                rms_val = float(rms)
            return rms_val < self._vad_rms_threshold

        def pop_segment_from_buffer() -> Optional[torch.Tensor]:
            nonlocal buffer_total
            if buffer_total < self._segment_samples:
                return None
            need = self._segment_samples
            take_list = []
            while need > 0 and buffer_chunks:
                chunk: torch.Tensor = buffer_chunks[0]
                if len(chunk) <= need:
                    take_list.append(chunk)
                    buffer_chunks.pop(0)
                    need -= len(chunk)
                else:
                    take_list.append(chunk[:need])
                    buffer_chunks[0] = chunk[need:]
                    need = 0
            seg = torch.cat(take_list) if len(take_list) > 1 else take_list[0]
            buffer_total -= len(seg)
            return seg

        try:
            while True:
                item = await audio_queue.get()
                if item is None:
                    # flush 余量
                    while buffer_total > 0:
                        seg = (
                            torch.cat(buffer_chunks)
                            if len(buffer_chunks) > 1
                            else (buffer_chunks[0] if buffer_chunks else torch.tensor([]))
                        )
                        buffer_chunks.clear()
                        buffer_total = 0
                        if seg.numel() > 0:
                            # 静音段直接跳过
                            if is_silent(seg):
                                logger.debug(f"[{session_id}] flush 段静音，跳过")
                            else:
                                t0 = time.perf_counter_ns()
                                tokens = transcriber.infer(seg, True)
                                t1 = time.perf_counter_ns()
                                if tokens.numel() > 0:
                                    text = transcriber.tokenizer.decode(tokens.tolist())
                                    await websocket.send(json.dumps({"type": "final", "text": text}, ensure_ascii=False))
                                    logger.info(f"flush cost: {(t1 - t0)/1e6:.2f} ms, text: {text}")
                    await websocket.send(json.dumps({"type": "done"}, ensure_ascii=False))
                    break

                # 正常帧
                audio_tensor = audio_to_tensor(item, sample_rate=SAMPLING_RATE)
                if audio_tensor.numel() == 0:
                    continue
                buffer_chunks.append(audio_tensor)
                buffer_total += len(audio_tensor)

                # 产出片段
                while True:
                    seg = pop_segment_from_buffer()
                    if seg is None:
                        break
                    # 静音段直接跳过
                    if is_silent(seg):
                        logger.debug(f"[{session_id}] 段静音，跳过")
                        continue
                    t0 = time.perf_counter_ns()
                    tokens = transcriber.infer(seg, False)
                    t1 = time.perf_counter_ns()
                    if tokens.numel() > 0:
                        text = transcriber.tokenizer.decode(tokens.tolist())
                        await websocket.send(json.dumps({"type": "partial", "text": text}, ensure_ascii=False))
                        logger.info(f"cost: {(t1 - t0)/1e6:.2f} ms, text: {text}")

        except Exception as e:
            logger.error(f"transcribe loop error: {session_id}: {e}")
            try:
                await websocket.send(json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False))
            except Exception:
                pass

    # 结果推送协程已移除，改为在转写协程中直接发送


