"""
核心数据类型和基础类
"""

from typing import List, Optional
import time

class ASRToken:
    """ASR Token类，表示转录结果中的一个词"""
    
    def __init__(self, start: float, end: float, text: str, 
                 probability: float = 0.95, speaker: int = -1, 
                 detected_language: Optional[str] = None):
        self.start = start
        self.end = end
        self.text = text
        self.probability = probability
        self.detected_language = detected_language
    
    def with_offset(self, offset: float):
        """添加时间偏移"""
        self.start += offset
        self.end += offset
        return self
    
    def __repr__(self):
        return f"ASRToken(text='{self.text}', start={self.start:.2f}, end={self.end:.2f}, speaker={self.speaker})"

class Transcript:
    """Transcript类，表示完整的转录结果"""
    
    def __init__(self, text: str = "", tokens: Optional[List[ASRToken]] = None):
        self.text = text
        self.tokens = tokens or []
    
    @classmethod
    def from_tokens(cls, tokens: List[ASRToken], sep: str = ''):
        """从tokens创建Transcript"""
        text = sep.join(token.text for token in tokens)
        return cls(text, tokens)
    
    def __repr__(self):
        return f"Transcript(text='{self.text}', tokens_count={len(self.tokens)})"



class AudioSession:
    """音频会话类，管理单个客户端的音频流"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.audio_buffer = []
        self.last_activity = time.time()
        self.is_active = True
        self.vad_processor = None
        self.transcription_processor = None
    
    def add_audio(self, audio_data: bytes):
        """添加音频数据"""
        self.audio_buffer.append(audio_data)
        self.last_activity = time.time()
    
    def get_audio_buffer(self) -> List[bytes]:
        """获取音频缓冲区"""
        return self.audio_buffer
    
    def clear_buffer(self):
        """清空音频缓冲区"""
        self.audio_buffer.clear()
    
    def is_session_active(self, timeout: float = 30.0) -> bool:
        """检查会话是否活跃"""
        return time.time() - self.last_activity < timeout
    
    def cleanup(self):
        """清理会话资源"""
        self.clear_buffer()
        self.is_active = False
        if self.vad_processor:
            del self.vad_processor
        if self.transcription_processor:
            del self.transcription_processor
    
    def __repr__(self):
        return f"AudioSession(id='{self.session_id}', active={self.is_active}, buffer_size={len(self.audio_buffer)})"

# 常量定义
PUNCTUATION_MARKS = {'.', '!', '?', ',', ';', ':'}
DEC_PAD = 50257
SAMPLING_RATE = 16000
