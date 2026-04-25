"""
测试客户端
"""

import asyncio
import websockets
import json
import base64
import logging
import numpy as np
import torch
import time
from whisper.simul_whisper.simul_whisper.transcriber.config import AlignAttConfig
from whisper.simul_whisper.simul_whisper.transcriber.simul_whisper import PaddedAlignAttWhisper, DEC_PAD
from whisper.simul_whisper.simul_whisper.transcriber.segment_loader import SegmentWrapper
from faster_whisper import WhisperModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_client():
    """测试客户端"""
    uri = "ws://localhost:8765"
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("连接到WebSocket服务器")
            
            # 接收连接确认
            response = await websocket.recv()
            data = json.loads(response)
            logger.info(f"服务器响应: {data}")
            
            # 生成测试音频
            duration = 3.0
            sample_rate = 16000
            t = torch.linspace(0, duration, int(sample_rate * duration))
            frequency = 440  # A4音符
            audio = 0.3 * torch.sin(2 * torch.pi * frequency * t)
            
            # 添加噪声
            noise = 0.05 * torch.randn_like(audio)
            audio = audio + noise
            
            # 转换为字节
            audio_bytes = (audio.numpy() * 32767).astype(np.int16).tobytes()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # 发送音频数据
            message = {
                "type": "audio",
                "data": audio_b64,
                "sample_rate": sample_rate,
                "channels": 1
            }
            
            await websocket.send(json.dumps(message))
            logger.info("音频数据已发送")
            
            # 接收转录结果
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                data = json.loads(response)
                logger.info(f"转录结果: {data}")
            except asyncio.TimeoutError:
                logger.warning("等待转录结果超时")
            
            # 发送ping
            ping_message = {"type": "ping"}
            await websocket.send(json.dumps(ping_message))
            
            # 接收pong
            response = await websocket.recv()
            data = json.loads(response)
            logger.info(f"Pong响应: {data}")
            
    except Exception as e:
        logger.error(f"客户端测试失败: {e}")

if __name__ == "__main__":
    segment_length = 1 # chunk length, in seconds
    frame_threshold = 12 # threshold for the attention-guided decoding, in frames
    buffer_len = 10 # the lengths for the context buffer, in seconds
    min_seg_len = 0.0 # transcibe only when the context buffer is larger than this threshold. Useful when the segment_length is small
    language = "en"

    audio_path = "/asr/live_stream_whisper/tests/english.wav"
    
    from whisper.simul_whisper.simul_whisper.transcriber.config import AlignAttConfig as TranscriberAlignAttConfig

    cfg = TranscriberAlignAttConfig(
        model_path="./models/base.pt", 
        language=language,
        segment_length=segment_length,
        min_seg_len=min_seg_len,
        if_ckpt_path="./whisper/simul_whisper/cif_models/base.pt"
    )
    # 先构造 PyTorch 模型，确定 device/dtype
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    compute_type = 'float16' if use_cuda else 'float32'
    fw_encoder = WhisperModel(
        "base",
        device=device,
        compute_type=compute_type,
    )
    segmented_audio = SegmentWrapper(audio_path=audio_path, segment_length=segment_length)
    model = PaddedAlignAttWhisper(cfg)


    hyp_list = []
    seg_cost_ms = []
    enc_cost_ms = []
    logger.info(f"start transcribe audio")
    text=""
    for seg_id, (seg, is_last) in enumerate(segmented_audio):
        logger.debug(f"seg_id: {seg_id} start infer,len:{len(seg)}")
        # 使用单调时钟统计毫秒耗时，避免系统时间回拨导致负值
        start_ns = time.perf_counter_ns()
        new_toks = model.infer(seg, is_last)
        hyp_list.append(new_toks)
        if model.last_encoder_time_ns is not None:
            enc_cost_ms.append(model.last_encoder_time_ns/1e6)
        hyp = torch.cat(hyp_list, dim=0)
        hyp = hyp[hyp < DEC_PAD]
        hyp = model.tokenizer.decode(hyp)
        cost = (time.perf_counter_ns() - start_ns) / 1e6
        logger.debug(f"seg_id: {seg_id} infer time: {cost} ms")
        seg_cost_ms.append(cost)
        text=hyp
        logger.info(f"seg_id: {seg_id} for {hyp}")

    model.refresh_segment(complete=True)
    logger.debug(f"text: {text}")
    if seg_cost_ms:
        avg_ms = sum(seg_cost_ms) / len(seg_cost_ms)
        logger.info(f"Average seg infer time: {avg_ms:.2f} ms over {len(seg_cost_ms)} segments")
    if enc_cost_ms:
        avg_enc_ms = sum(enc_cost_ms) / len(enc_cost_ms)
        logger.info(f"Average encoder time: {avg_enc_ms:.2f} ms over {len(enc_cost_ms)} segments")
