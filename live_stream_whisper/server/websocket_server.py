import asyncio
import json
import logging
import uuid
from typing import Dict
import time
import websockets
from websockets.server import WebSocketServerProtocol

from config.config import config_manager
from core.types import SAMPLING_RATE
from core.transcriber import Transcriber


logger = logging.getLogger(__name__)


class WebSocketTranscriptionServer:
    def __init__(self, host: str = "localhost", port: int = 8765):
        cfg = config_manager.get_config()
        self._server_cfg = cfg["server"]

        self.host = host or self._server_cfg.host
        self.port = port or self._server_cfg.port

        self._connections: Dict[str, Dict[str, object]] = {}
        self._stop_event = asyncio.Event()


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
            await self._stop_event.wait()

    async def stop(self):
        try:
            self._stop_event.set()
        except Exception:
            pass

    async def _handler(self, websocket: WebSocketServerProtocol):
        session_id = str(uuid.uuid4())
        audio_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

        self._connections[session_id] = {
            "websocket": websocket,
            "audio_queue": audio_queue,
        }

        logger.info(f"新连接: {session_id}")


       # 接收音頻的task
        producer_task = asyncio.create_task(
            self._receive_audio_loop(session_id, websocket, audio_queue)
        )

        # 组件加载
        transcriber = Transcriber()      

        # 在后台线程加载模型，避免阻塞事件循环导致消息无法发出
        try:
            await asyncio.to_thread(transcriber.load)
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            try:
                await websocket.send(json.dumps({"type": "error", "message": "model load failed"}, ensure_ascii=False))
            except Exception:
                pass
            return

        # ready gating（模型加载完成后再发送ready）
        try:
            await websocket.send(json.dumps({
                "type": "ready",
                "session_id": session_id,
                "sample_rate": SAMPLING_RATE,
            }, ensure_ascii=False))
        except Exception:
            pass

        consumer_task = asyncio.create_task(
        transcriber.process_audio(session_id, websocket, audio_queue)
        )

        try:
            done, pending = await asyncio.wait({producer_task, consumer_task}, return_when=asyncio.FIRST_EXCEPTION)
            for t in done:
                exc = t.exception()
                if exc:
                    raise exc
        except Exception as e:
            logger.error(f"会话 {session_id} 发生错误: {e}")
        finally:
            try:
                await audio_queue.put(None)
            except Exception:
                pass
            for task in (producer_task, consumer_task):
                if not task.done():
                    task.cancel()
            self._connections.pop(session_id, None)
            logger.info(f"连接关闭: {session_id}")


    async def _receive_audio_loop(self, session_id: str, websocket: WebSocketServerProtocol, audio_queue: asyncio.Queue):
        try:
            async for message in websocket:
                if isinstance(message, (bytes, bytearray)):
                    try:
                        chunk={
                            "beg":time.perf_counter_ns(),
                            "data":bytes(message)
                        }
                        await audio_queue.put(chunk)
                    except Exception:
                        logger.warning(f"音频队列写入失败，丢弃一帧: {session_id}")
                elif isinstance(message, str):
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
        except Exception as e:
            logger.warning(f"接收端异常 {session_id}: {e}")
        finally:
            try:
                await audio_queue.put(None)
            except Exception:
                pass




