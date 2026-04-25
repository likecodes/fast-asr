import logging
import time
from typing import Optional

import torch
import re
import json
import asyncio
from websockets.server import WebSocketServerProtocol

from utils.tools import audio_to_tensor
from engines.vad.vad import VAD

from core.types import SAMPLING_RATE
from config.config import config_manager
from engines.simul_whisper.transcriber import AlignAttConfig, PaddedAlignAttWhisper


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Transcriber:
    def __init__(self):
        cfg = config_manager.get_whisper_config()

        model_path = cfg.model_path
        language = cfg.language
        frame_threshold = cfg.frame_threshold
        nonspeech_prob = cfg.nonspeech_prob
        rewind_threshold = cfg.rewind_threshold
        segment_length_s = cfg.segment_length
        buffer_length_s = cfg.audio_max_len
        min_seg_length_s=cfg.min_seg_length
        cif_ckpt_path = cfg.cif_ckpt_path or ''
        static_prompt = cfg.static_init_prompt

        self.align_cfg = AlignAttConfig(
            model_path=model_path,
            segment_length=segment_length_s,
            min_seg_len=min_seg_length_s,
            buffer_len=buffer_length_s,
            language=language,
            prompt=static_prompt,
            frame_threshold=frame_threshold,
            nonspeech_prob=nonspeech_prob,
            rewind_threshold=rewind_threshold,
            if_ckpt_path=cif_ckpt_path or "",
        )
        logger.info(f"align_cfg: {self.align_cfg}")
        self.segment_samples = int(self.align_cfg.segment_length * SAMPLING_RATE)
        self.model: Optional[PaddedAlignAttWhisper] = None
        self.vad: Optional[VAD] = None
        self.infer_cnt=0
        self.total_time_cost_ns = 0

    def load(self):
        logger.info("loading model")
        self.model = PaddedAlignAttWhisper(self.align_cfg)
        try:
            warmup_seg = torch.zeros(int(0.1 * SAMPLING_RATE))
            # _ = self.model.infer(warmup_seg, True)
        except Exception:
            pass
        try:
            self.vad = VAD()
        except Exception:
            self.vad = None

    def infer_segment(self, seg: torch.Tensor, is_last: bool) -> (str):
        tokens = self.model.infer(seg, is_last)
        if tokens.numel() > 0:
            return self.model.tokenizer.decode(tokens.tolist())
        else:
            text = ""
        return text

    async def process_audio(self, session_id: str, websocket: WebSocketServerProtocol, audio_queue: asyncio.Queue):
        try:
            while True:
                item = await audio_queue.get()
                if item is None or item["data"] is None:
                    await websocket.send(json.dumps({"type": "done"}, ensure_ascii=False))
                    break
                audioChunk = item["data"]
                seg = audio_to_tensor(audioChunk, sample_rate=SAMPLING_RATE)
                if seg.numel() == 0:
                    continue

                # 64ms 小块静音快速跳过
                if self.vad and self.vad.is_chunk_silent(seg):
                    logger.debug(f"[{session_id}] 64ms 静音，跳过")
                    continue

                text = self.infer_segment(seg, False)
                if text:
                    await websocket.send(json.dumps({"type": "partial", "text": text}, ensure_ascii=False))
                    end = time.perf_counter_ns()
                    current_cost_ns = end-item["beg"]
                    self.infer_cnt += 1
                    self.total_time_cost_ns += current_cost_ns
                    logger.info(f"transcribe current inter time cost:{current_cost_ns/1e6:.2f} ms, avg time cost:{self.total_time_cost_ns / self.infer_cnt / 1e6:2f} ms")

        except Exception as e:
            logger.error(f"transcribe loop error: {session_id}: {e}")
            try:
                await websocket.send(json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False))
            except Exception:
                pass

