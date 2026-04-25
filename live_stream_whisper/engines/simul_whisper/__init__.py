"""simul_whisper engine adapter.

This package re-exports classes from the legacy path
`whisper.simul_whisper.simul_whisper` so callers can migrate to
`engines.simul_whisper` without moving the original sources immediately.
"""

from .transcriber import AlignAttConfig, PaddedAlignAttWhisper  # re-export

__all__ = [
    'AlignAttConfig',
    'PaddedAlignAttWhisper',
]


