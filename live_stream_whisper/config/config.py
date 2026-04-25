"""
合并配置模块：原 core.config 与 config.align_config 统一至此。
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class WhisperConfig:
    model_path: str = "./models/medium.pt"
    language: str = "auto"
    task: str = "transcribe"
    beam_size: int = 1
    decoder_type: str = "beam"
    frame_threshold: int = 25
    audio_max_len: float = 20.0
    audio_min_len: float = 0.0
    segment_length: float = 0.064
    min_seg_length: float = 0.0
    cif_ckpt_path: Optional[str] = None
    nonspeech_prob: float = 0.5
    rewind_threshold: int = 5
    max_context_tokens: Optional[int] = None
    init_prompt: Optional[str] = None
    static_init_prompt: Optional[str] = None


@dataclass
class VADConfig:
    enabled: bool = True
    threshold: float = 0.5
    min_silence_duration: float = 0.1
    min_speech_duration: float = 0.1
    window_size: int = 1600
    stride: int = 160
    model_path: str = "silero_vad"
    speech_frames_to_infer: int = 5



@dataclass
class ServerConfig:
    host: str = "localhost"
    port: int = 8765
    max_size: int = 2**20
    ping_interval: int = 30
    ping_timeout: int = 10


class ConfigManager:
    def __init__(self):
        self._config = None
        self._load_config()

    def _load_config(self):
        self._config = {
            'whisper': WhisperConfig(
                model_path=os.getenv('WHISPER_MODEL_PATH', './models/whisper/base.pt'),
                language=os.getenv('WHISPER_LANGUAGE', 'en'),
                task=os.getenv('WHISPER_TASK', 'transcribe'),
                beam_size=int(os.getenv('WHISPER_BEAM_SIZE', '3')),
                decoder_type=os.getenv('WHISPER_DECODER_TYPE', 'beam'),
                frame_threshold=int(os.getenv('WHISPER_FRAME_THRESHOLD', '4')),
                audio_max_len=float(os.getenv('WHISPER_AUDIO_MAX_LEN', '5.0')),
                audio_min_len=float(os.getenv('WHISPER_AUDIO_MIN_LEN', '0.0')),
                segment_length=float(os.getenv('WHISPER_SEGMENT_LENGTH', '0.064')),
                min_seg_length=float(os.getenv('WHISPER_MIN_SEG_LENGTH', '0.0')),
                cif_ckpt_path=os.getenv('WHISPER_CIF_CKPT_PATH','./models/cif_models/base.pt'),
                nonspeech_prob=float(os.getenv('WHISPER_NONSPEECH_PROB', '0.5')),
                rewind_threshold=int(os.getenv('WHISPER_REWIND_THRESHOLD', '5')),
                max_context_tokens=int(os.getenv('WHISPER_MAX_CONTEXT_TOKENS', '0')) or None,
                init_prompt=os.getenv('WHISPER_INIT_PROMPT'),
                # static_init_prompt=os.getenv('WHISPER_STATIC_INIT_PROMPT',"以下普通话的句子."),
            ),
            'vad': VADConfig(
                enabled=os.getenv('VAD_ENABLED', 'true').lower() == 'true',
                threshold=float(os.getenv('VAD_THRESHOLD', '0.5')),
                min_silence_duration=float(os.getenv('VAD_MIN_SILENCE_DURATION', '0.1')),
                min_speech_duration=float(os.getenv('VAD_MIN_SPEECH_DURATION', '0.1')),
                window_size=int(os.getenv('VAD_WINDOW_SIZE', '512')),
                stride=int(os.getenv('VAD_STRIDE', '160')),
                model_path=os.getenv('VAD_MODEL_PATH', 'silero_vad'),
                speech_frames_to_infer=int(os.getenv('VAD_SPEECH_FRAMES_TO_INFER', '10')),
            ),
            'server': ServerConfig(
                host=os.getenv('SERVER_HOST', 'localhost'),
                port=int(os.getenv('SERVER_PORT', '8765')),
                max_size=int(os.getenv('SERVER_MAX_SIZE', str(2**20))),
                ping_interval=int(os.getenv('SERVER_PING_INTERVAL', '30')),
                ping_timeout=int(os.getenv('SERVER_PING_TIMEOUT', '10')),
            )
        }

        logger.info("配置加载完成")

    def get_config(self) -> Dict[str, Any]:
        return self._config

    def get_whisper_config(self) -> WhisperConfig:
        return self._config['whisper']

    def get_vad_config(self) -> VADConfig:
        return self._config['vad']

    def get_server_config(self) -> ServerConfig:
        return self._config['server']

    def update_config(self, section: str, **kwargs):
        if section in self._config:
            config_obj = self._config[section]
            for key, value in kwargs.items():
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
                    logger.info(f"更新配置 {section}.{key} = {value}")
                else:
                    logger.warning(f"配置项 {section}.{key} 不存在")
        else:
            logger.error(f"配置节 {section} 不存在")


config_manager = ConfigManager()


