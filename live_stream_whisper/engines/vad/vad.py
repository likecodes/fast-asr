import logging
from typing import Optional

import torch

from config.config import config_manager
from core.types import SAMPLING_RATE
from .silero_vad_iterator import FixedVADIterator


logger = logging.getLogger(__name__)


class VAD:
    def __init__(self):
        cfg = config_manager.get_vad_config()
        self.threshold = float(getattr(cfg, 'threshold', 0.5))
        self.min_silence_ms = int(getattr(cfg, 'min_silence_duration', 0.1) * 1000)
        self.speech_pad_ms = int(getattr(cfg, 'min_speech_duration', 0.1) * 1000)

        self.model = None
        self.iterator: Optional[FixedVADIterator] = None
        try:
            self.model, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
            logger.info(f"Silero VAD 加载成功 threshold={self.threshold}")
            self.iterator = FixedVADIterator(
                self.model,
                threshold=self.threshold,
                sampling_rate=SAMPLING_RATE,
                min_silence_duration_ms=self.min_silence_ms,
                speech_pad_ms=self.speech_pad_ms,
            )
        except Exception as e:
            self.model = None
            self.iterator = None
            logger.warning(f"Silero VAD 加载失败")

    def analyze(self, audio_tensor: torch.Tensor):
        """直接透传 FixedVADIterator 的返回（可能为 None 或包含 start/end 的 dict）。"""
        if audio_tensor is None or audio_tensor.numel() == 0:
            return None
        if self.iterator is None:
            return None
        try:
            return self.iterator(audio_tensor, return_seconds=False)
        except Exception:
            return None

    def is_triggered(self) -> bool:
        try:
            return bool(getattr(self.iterator, 'triggered', False))
        except Exception:
            return False

    def is_chunk_silent(self, audio_tensor: torch.Tensor) -> bool:
        """逐512采样窗检测该块是否完全静音（所有窗的 speech_prob 都低于 threshold）。

        说明：仅用于小块（如 ~64ms, 1024 样本）的快速 gating，不依赖 iterator 状态。
        """
        if audio_tensor is None or audio_tensor.numel() == 0:
            return True
        if self.model is None:
            # 没有模型时不做强判，避免误丢；可按需改成 True
            return False
        x = audio_tensor
        if x.dim() != 1:
            x = x.view(-1)
        win = 512
        total = len(x)
        pos = 0
        while pos < total:
            frame = x[pos:pos+win]
            if len(frame) < win:
                # 右侧用零填充到512
                pad = torch.zeros(win - len(frame), dtype=frame.dtype, device=frame.device)
                frame = torch.cat([frame, pad], dim=0)
            frame_in = frame.unsqueeze(0)
            try:
                prob = float(self.model(frame_in, SAMPLING_RATE).item())
            except Exception:
                # 模型异常时保守认为非静音，避免误丢
                return False
            if prob >= self.threshold:
                return False
            pos += win
        return True

    # 保留触发状态查询，供上层决策参考


