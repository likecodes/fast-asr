"""
主模块初始化
"""

# 导入核心模块
from .core import (
    ASRToken,
    Transcript,
    ChangeSpeaker,
    AudioSession,
    config_manager,
    WhisperConfig,
    VADConfig,
    TaskConfig,
    ServerConfig
)

# 导入Whisper模块
from .whisper import (
    EnhancedPaddedAlignAttWhisper,
    AudioStreamWhisper
)

# 导入VAD模块
from .vad import (
    VADIterator,
    FixedVADIterator,
    VADProcessor,
    SileroVADProcessor,
    EnhancedVADProcessor,
    VADManager
)

# 导入服务器模块
from .server import (
    MultiTaskProcessor,
    TaskPriority,
    TaskStatus,
    Task,
    WebSocketTranscriptionServer
)

# 导入接口模块
from .core.transcription_base import (
    TranscriptionBase,
    create_transcription_factory
)

# 导入工具模块
from .utils import (
    generate_test_audio,
    audio_to_tensor,
    tensor_to_audio,
    format_timestamp,
    calculate_audio_duration,
    resample_audio,
    validate_audio_format,
    PerformanceTimer,
    retry_async,
    get_memory_usage,
    log_system_info
)

# 导入测试模块
from .tests import (
    test_enhanced_backend,
    test_configuration,
    test_faster_whisper_support,
    test_vad_processor,
    test_multi_task_processor,
    run_all_tests
)

__version__ = "2.0.0"
__author__ = "Leo.he"

__all__ = [
    # 核心模块
    'ASRToken',
    'Transcript',
    'ChangeSpeaker',
    'AudioSession',
    'config_manager',
    'WhisperConfig',
    'VADConfig',
    'TaskConfig',
    'ServerConfig',
    
    # Whisper模块
    'EnhancedPaddedAlignAttWhisper',
    'AudioStreamWhisper',
    
    # VAD模块
    'VADIterator',
    'FixedVADIterator',
    'VADProcessor',
    'SileroVADProcessor',
    'EnhancedVADProcessor',
    'VADManager',
    
    # 服务器模块
    'MultiTaskProcessor',
    'TaskPriority',
    'TaskStatus',
    'Task',
    'WebSocketTranscriptionServer',
    
    # 接口模块
    'TranscriptionBase',
    'create_transcription_factory',
    
    # 工具模块
    'generate_test_audio',
    'audio_to_tensor',
    'tensor_to_audio',
    'format_timestamp',
    'calculate_audio_duration',
    'resample_audio',
    'validate_audio_format',
    'PerformanceTimer',
    'retry_async',
    'get_memory_usage',
    'log_system_info',
    
    # 测试模块
    'test_enhanced_backend',
    'test_configuration',
    'test_faster_whisper_support',
    'test_vad_processor',
    'test_multi_task_processor',
    'run_all_tests'
]