import jsonlines

import torch
import torchaudio
from torchaudio.functional import resample as ta_resample
import numpy as np
import wave
try:
    import soundfile as sf
except Exception:
    sf = None

from ..whisper.audio import N_FFT, HOP_LENGTH, SAMPLE_RATE
from .config import AlignAttConfig


class Segment:
    def __init__(self, audio_path, samples_to_read, samples_in_chunk):
        self.audio_path = audio_path
        # Robust loader: try torchaudio first, fallback to soundfile
        try:
            audio, sr = torchaudio.load(audio_path, normalize=True)
        except Exception:
            data = None
            sr = None
            # Try soundfile first if available
            if sf is not None:
                try:
                    data, sr = sf.read(audio_path, dtype='float32', always_2d=False)
                except Exception:
                    data = None
            # Fallback to built-in wave for PCM WAV
            if data is None:
                try:
                    with wave.open(audio_path, 'rb') as wf:
                        sr = wf.getframerate()
                        n_channels = wf.getnchannels()
                        n_frames = wf.getnframes()
                        pcm = wf.readframes(n_frames)
                        dtype = np.int16 if wf.getsampwidth() == 2 else np.int8
                        arr = np.frombuffer(pcm, dtype=dtype).astype('float32')
                        if n_channels > 1:
                            arr = arr.reshape(-1, n_channels).mean(axis=1)
                        data = (arr / (32768.0 if dtype == np.int16 else 128.0))
                except Exception:
                    data = None
            if data is None or sr is None:
                # Re-raise original error path
                raise
            audio = torch.from_numpy(data).unsqueeze(0)
        # Resample if needed
        if sr != SAMPLE_RATE:
            audio = ta_resample(audio, sr, SAMPLE_RATE)
        self.audio = audio.squeeze()
        self.audio_len_s = self.audio.shape[0] / SAMPLE_RATE
        self.samples_to_read = samples_to_read
        self.samples_in_chunk = samples_in_chunk
        self.buffer_len = samples_in_chunk - samples_to_read

    def __iter__(self):
        frames_in_chunk = self.audio[:self.samples_in_chunk]
        read_pointer = frames_in_chunk.shape[0]
        yield frames_in_chunk, (read_pointer >= self.audio.shape[0])
        while read_pointer < self.audio.shape[0]:
            frames_in_chunk = torch.cat(
                (frames_in_chunk[-self.buffer_len:], self.audio[read_pointer:read_pointer+self.samples_to_read]),
                dim=0
                )
            read_pointer += self.samples_to_read
            yield frames_in_chunk, (read_pointer >= self.audio.shape[0])


class SegmentWrapper(Segment):
    def __init__(self, audio_path, segment_length):
        frames_to_read = int((segment_length * SAMPLE_RATE) / HOP_LENGTH)
        samples_to_read = frames_to_read * HOP_LENGTH
        samples_in_chunk = samples_to_read + N_FFT - HOP_LENGTH
        super().__init__(
            audio_path, 
            samples_to_read=samples_to_read, 
            samples_in_chunk=samples_in_chunk)


class SegmentLoader:

    def __init__(self, cfg: AlignAttConfig):
        self.cfg = cfg
        frames_to_read = int((cfg.segment_length * SAMPLE_RATE) / HOP_LENGTH)
        self.samples_to_read = frames_to_read * HOP_LENGTH
        self.samples_in_chunk = self.samples_to_read + N_FFT - HOP_LENGTH
        with open(cfg.eval_data_path) as f:
            self.data_list = [l for l in jsonlines.Reader(f)]
    
    def __getitem__(self, i):
        return Segment(
            audio_path=self.data_list[i]["audio"], 
            samples_to_read=self.samples_to_read,
            samples_in_chunk=self.samples_in_chunk
        ), self.data_list[i]["sentence"]
    
    def __len__(self) -> int:
        return len(self.data_list)
    