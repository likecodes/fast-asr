class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._target = 16000;
    this._src = sampleRate;
  }

  _resampleFloat32(input) {
    if (!input) return new Float32Array();
    if (this._src === this._target) return input;
    const ratio = this._target / this._src;
    const newLen = Math.floor(input.length * ratio);
    const out = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {
      const t = i / ratio;
      const i0 = Math.floor(t);
      const i1 = Math.min(i0 + 1, input.length - 1);
      const frac = t - i0;
      out[i] = input[i0] * (1 - frac) + input[i1] * frac;
    }
    return out;
  }

  _floatTo16BitPCM(float32) {
    const len = float32.length;
    const out = new Int16Array(len);
    for (let i = 0; i < len; i++) {
      let s = Math.max(-1, Math.min(1, float32[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return out;
  }

  process(inputs) {
    const input = inputs[0];
    const ch0 = input && input[0];
    if (!ch0) return true;
    const res = this._resampleFloat32(ch0);
    const pcm16 = this._floatTo16BitPCM(res);
    this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
    return true;
  }
}

registerProcessor('pcm-processor', PCMProcessor);


