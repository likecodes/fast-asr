"""
测试模块初始化
"""

from .test_modules import (
    test_enhanced_backend,
    test_configuration,
    test_faster_whisper_support,
    test_vad_processor,
    test_multi_task_processor,
    main as run_all_tests
)

__all__ = [
    'test_enhanced_backend',
    'test_configuration',
    'test_faster_whisper_support',
    'test_vad_processor',
    'test_multi_task_processor',
    'run_all_tests'
]
