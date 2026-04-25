"""
工具函数
"""

import logging
import time
import asyncio
from typing import Optional, Dict, Any
import torch
import numpy as np

logger = logging.getLogger(__name__)

def generate_test_audio(duration: float = 3.0, sample_rate: int = 16000) -> torch.Tensor:
    """生成测试音频"""
    # 生成简单的正弦波作为测试音频
    t = torch.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4音符
    audio = 0.3 * torch.sin(2 * torch.pi * frequency * t)
    
    # 添加一些噪声
    noise = 0.05 * torch.randn_like(audio)
    audio = audio + noise
    
    return audio

def audio_to_tensor(audio_data: bytes, sample_rate: int = 16000) -> torch.Tensor:
    """将音频字节数据转换为tensor"""
    try:
        # 假设是16位PCM数据
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        return torch.from_numpy(audio_array)
    except Exception as e:
        logger.error(f"音频转换失败: {e}")
        return torch.tensor([])

def tensor_to_audio(audio_tensor: torch.Tensor, sample_rate: int = 16000) -> bytes:
    """将tensor转换为音频字节数据"""
    try:
        # 转换为16位PCM
        audio_array = (audio_tensor.numpy() * 32767).astype(np.int16)
        return audio_array.tobytes()
    except Exception as e:
        logger.error(f"音频转换失败: {e}")
        return b""

def format_timestamp(seconds: float) -> str:
    """格式化时间戳"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def calculate_audio_duration(audio_tensor: torch.Tensor, sample_rate: int = 16000) -> float:
    """计算音频时长"""
    return len(audio_tensor) / sample_rate

def resample_audio(audio_tensor: torch.Tensor, original_rate: int, target_rate: int) -> torch.Tensor:
    """重采样音频"""
    if original_rate == target_rate:
        return audio_tensor
    
    try:
        import torchaudio
        # 使用torchaudio进行重采样
        resampler = torchaudio.transforms.Resample(original_rate, target_rate)
        return resampler(audio_tensor)
    except ImportError:
        logger.warning("torchaudio未安装，使用简单重采样")
        # 简单的线性插值重采样
        ratio = target_rate / original_rate
        new_length = int(len(audio_tensor) * ratio)
        indices = torch.linspace(0, len(audio_tensor) - 1, new_length)
        return torch.interp(indices, torch.arange(len(audio_tensor)), audio_tensor)

def validate_audio_format(audio_tensor: torch.Tensor, sample_rate: int = 16000) -> bool:
    """验证音频格式"""
    if not isinstance(audio_tensor, torch.Tensor):
        logger.error("音频数据不是torch.Tensor")
        return False
    
    if audio_tensor.dim() != 1:
        logger.error("音频数据不是一维tensor")
        return False
    
    if len(audio_tensor) == 0:
        logger.error("音频数据为空")
        return False
    
    duration = calculate_audio_duration(audio_tensor, sample_rate)
    if duration > 30:  # 超过30秒
        logger.warning(f"音频时长过长: {duration:.2f}秒")
        return False
    
    return True

class PerformanceTimer:
    """性能计时器"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"{self.name} 耗时: {duration:.3f}秒")
    
    def elapsed(self) -> Optional[float]:
        """获取已用时间"""
        if self.start_time is None:
            return None
        end_time = self.end_time or time.time()
        return end_time - self.start_time

async def retry_async(func, max_retries: int = 3, delay: float = 1.0, *args, **kwargs):
    """异步重试装饰器"""
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"重试失败，已达到最大重试次数: {e}")
                raise
            
            logger.warning(f"重试 {attempt + 1}/{max_retries}: {e}")
            await asyncio.sleep(delay * (2 ** attempt))  # 指数退避

def get_memory_usage() -> Dict[str, float]:
    """获取内存使用情况"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss / 1024 / 1024,  # MB
            "vms": memory_info.vms / 1024 / 1024,  # MB
            "percent": process.memory_percent()
        }
    except ImportError:
        logger.warning("psutil未安装，无法获取内存使用情况")
        return {"rss": 0, "vms": 0, "percent": 0}

def log_system_info():
    """记录系统信息"""
    try:
        import torch
        import platform
        
        logger.info(f"Python版本: {platform.python_version()}")
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA版本: {torch.version.cuda}")
            logger.info(f"GPU数量: {torch.cuda.device_count()}")
        
        memory_info = get_memory_usage()
        logger.info(f"内存使用: {memory_info['rss']:.1f}MB ({memory_info['percent']:.1f}%)")
        
    except Exception as e:
        logger.error(f"获取系统信息失败: {e}")
