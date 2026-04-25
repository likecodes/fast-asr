(() => {
  const $ = (sel) => document.querySelector(sel);

  const log = (msg) => {
    const el = $('#logs');
    el.textContent += `[${new Date().toLocaleTimeString()}] ${msg}\n`;
    el.scrollTop = el.scrollHeight;
  };

  const wsUrlInput = $('#ws-url');
  const btnConnect = $('#btn-connect');
  const btnDisconnect = $('#btn-disconnect');
  const btnRecord = $('#btn-record');
  const statusEl = $('#status');
  const finalEl = $('#final-text');
  const listeningEl = $('#listening');
  const caretEl = $('#caret');
  const errorMessageEl = $('#errorMessage');
  const emptyStateEl = $('#emptyState');
  const audioVisualizerEl = $('#audioVisualizer');
  const clearBtn = $('#clearBtn');
  const clearLogsBtn = $('#clearLogsBtn');

  let ws = null;
  let audioContext = null;
  let mediaStream = null;
  let scriptNode = null;
  let isCapturing = false;
  let serverReady = false;
  let recordingStartTime = 0;
  let recordingTimer = null;
  let audioDataSent = 0;
  let visualizerBars = [];
  let listenTimer = null;

  const BTN_START_TEXT = 'Start Recording';
  const BTN_START_INIT_TEXT = 'Initializing...';
  const TARGET_SR = 16000;

  function setRecordButtonState(state) {
    btnRecord.classList.remove('ready', 'recording');
    if (state === 'ready') {
      btnRecord.classList.add('ready');
      btnRecord.title = 'Start Recording';
    } else if (state === 'recording') {
      btnRecord.classList.add('recording');
      btnRecord.title = 'Stop Recording';
    }
  }

  function initVisualizer() {
    for (let i = 0; i < 60; i++) {
      const bar = document.createElement('div');
      bar.className = 'visualizer-bar';
      bar.style.height = '10px';
      audioVisualizerEl.appendChild(bar);
      visualizerBars.push(bar);
    }
  }

  function showError(message) {
    errorMessageEl.textContent = message;
    errorMessageEl.classList.add('show');
    setTimeout(() => errorMessageEl.classList.remove('show'), 5000);
    log(`Error: ${message}`);
  }

  function updateStatus(text, className) {
    statusEl.textContent = text;
    statusEl.className = `status-dot ${className}`;
  }

  function setConnected(connected) {
    if (connected) {
      btnConnect.style.display = 'none';
      btnDisconnect.style.display = 'inline-block';
      btnDisconnect.disabled = false;
      updateStatus('', 'connected');
      btnRecord.disabled = !serverReady;
    } else {
      btnConnect.style.display = 'inline-block';
      btnDisconnect.style.display = 'none';
      btnConnect.disabled = false;
      updateStatus('', 'disconnected');
      btnRecord.disabled = true;
    }
  }

  function updateStats() {
    if (isCapturing && recordingStartTime > 0) {
      const elapsed = (Date.now() - recordingStartTime) / 1000;
      $('#recordingTime').textContent = elapsed.toFixed(1) + 's';
    }

    $('#audioSent').textContent = (audioDataSent / 1024).toFixed(1) + ' KB';

    const wordCount = (finalEl.textContent || '').length;
    $('#wordCount').textContent = wordCount;
  }

  function updateVisualizer(audioData) {
    const samples = visualizerBars.length;
    const step = Math.floor(audioData.length / samples);

    for (let i = 0; i < samples && i < visualizerBars.length; i++) {
      const value = Math.abs(audioData[i * step] || 0);
      const height = Math.max(10, Math.min(60, value * 300));
      visualizerBars[i].style.height = height + 'px';
    }
  }

  function connect() {
    const url = wsUrlInput.value.trim();
    if (!url) {
      showError('Please enter WebSocket address');
      return;
    }

    try {
      ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        serverReady = false;
        setConnected(true);
        log('WebSocket connected, waiting for server ready...');
      };

      ws.onclose = () => {
        serverReady = false;
        setConnected(false);
        log('WebSocket closed');
        hideCaret();
        hideListening();
      };

      ws.onerror = () => {
        showError('WebSocket connection error');
      };

      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data.type === 'ready') {
            serverReady = true;
            btnRecord.disabled = false;
            setRecordButtonState('ready');
            setConnected(true);
            log(`Server ready, session=${data.session_id}`);
            hideListening();
          } else if (data.type === 'partial') {
            appendFinal(String(data.text || ''));
            hideListening();
            moveCaretToEnd();
            updateStats();
          } else if (data.type === 'final') {
            appendFinal(String(data.text || ''));
            hideListening();
            moveCaretToEnd();
            updateStats();
          } else if (data.type === 'done') {
            log('Server finished transcription');
            hideListening();
          } else if (data.type === 'error') {
            showError(`Server error: ${data.message || ''}`);
            hideListening();
          }
        } catch (_) {
        }
      };
    } catch (e) {
      showError('Connection failed: ' + e.message);
    }
  }

  function disconnect() {
    stopCapture();
    if (ws) {
      try { ws.close(); } catch (_) { }
      ws = null;
    }
    serverReady = false;
    setConnected(false);
  }

  async function startCapture() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      showError('Please connect to WebSocket first');
      return;
    }
    if (!serverReady) {
      showError('Server not ready, please wait for initialization');
      return;
    }
    if (isCapturing) return;

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        },
        video: false
      });

      audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
      });
      const source = audioContext.createMediaStreamSource(mediaStream);

      const BUFFER_SIZE = 1024;
      scriptNode = audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1);

      scriptNode.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);

        updateVisualizer(input);

        const pcm16 = floatTo16BitPCM(input);

        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(pcm16.buffer);
          audioDataSent += pcm16.buffer.byteLength;
        }

        scheduleListening();
      };

      source.connect(scriptNode);
      scriptNode.connect(audioContext.destination);

      isCapturing = true;
      recordingStartTime = Date.now();
      audioDataSent = 0;

      updateStatus('', 'recording');
      setRecordButtonState('recording');
      btnDisconnect.disabled = true;
      audioVisualizerEl.style.display = 'flex';
      emptyStateEl.style.display = 'none';

      showCaret();

      recordingTimer = setInterval(() => updateStats(), 100);

      log('Recording started');
    } catch (e) {
      showError('Cannot access microphone: ' + e.message);
    }
  }

  function stopCapture() {
    if (!isCapturing) return;

    try {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'stop' }));
      }

      if (scriptNode) {
        scriptNode.disconnect();
        scriptNode.onaudioprocess = null;
        scriptNode = null;
      }
      if (audioContext) {
        try { audioContext.close(); } catch (_) { }
        audioContext = null;
      }
      if (mediaStream) {
        mediaStream.getTracks().forEach(t => t.stop());
        mediaStream = null;
      }

      if (recordingTimer) {
        clearInterval(recordingTimer);
        recordingTimer = null;
      }
    } catch (_) { }

    isCapturing = false;

    updateStatus('', 'connected');
    setRecordButtonState('ready');
    btnDisconnect.disabled = false;
    audioVisualizerEl.style.display = 'none';

    hideCaret();
    hideListening();

    if (!finalEl.textContent && emptyStateEl) {
      emptyStateEl.style.display = 'block';
    }

    log('Recording stopped');
  }

  function floatTo16BitPCM(float32) {
    const len = float32.length;
    const out = new Int16Array(len);
    for (let i = 0; i < len; i++) {
      let s = Math.max(-1, Math.min(1, float32[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return out;
  }

  function appendFinal(text) {
    if (!text) return;
    if (emptyStateEl) {
      emptyStateEl.style.display = 'none';
    }
    finalEl.textContent = (finalEl.textContent + (finalEl.textContent ? ' ' : '') + text).trim();
    moveCaretToEnd();
  }

  function scheduleListening() {
    if (listenTimer) return;
    if (!isCapturing) return;

    listenTimer = setTimeout(() => {
      if (listeningEl && caretEl && caretEl.parentNode) {
        caretEl.insertAdjacentElement('afterend', listeningEl);
      }
      listeningEl.classList.remove('hidden');
      listenTimer = null;
    }, 6000);
  }

  function hideListening() {
    if (listenTimer) {
      clearTimeout(listenTimer);
      listenTimer = null;
    }
    if (listeningEl && !listeningEl.classList.contains('hidden')) {
      listeningEl.classList.add('hidden');
    }
  }

  function moveCaretToEnd() {
    if (!caretEl) return;
    finalEl.insertAdjacentElement('afterend', caretEl);
    if (!listeningEl.classList.contains('hidden')) {
      caretEl.insertAdjacentElement('afterend', listeningEl);
    }
    const transcript = document.getElementById('transcript');
    if (transcript) transcript.scrollTop = transcript.scrollHeight;
  }

  function showCaret() {
    if (caretEl) caretEl.classList.remove('hidden');
    moveCaretToEnd();
  }

  function hideCaret() {
    if (caretEl) caretEl.classList.add('hidden');
  }

  function clearResults() {
    finalEl.textContent = '';
    if (!isCapturing && emptyStateEl) {
      emptyStateEl.style.display = 'block';
    }
    audioDataSent = 0;
    $('#recordingTime').textContent = '0s';
    $('#wordCount').textContent = '0';
    $('#audioSent').textContent = '0 KB';
    updateStats();
    log('Transcription cleared');
  }

  function clearLogs() {
    $('#logs').textContent = '';
  }

  function init() {
    initVisualizer();
    hideCaret();
    hideListening();

    btnConnect.addEventListener('click', connect);
    btnDisconnect.addEventListener('click', disconnect);
    btnRecord.addEventListener('click', () => {
      if (isCapturing) {
        stopCapture();
      } else {
        startCapture();
      }
    });
    clearBtn.addEventListener('click', clearResults);
    clearLogsBtn.addEventListener('click', clearLogs);

    log('Application initialized');
  }

  init();
})();
