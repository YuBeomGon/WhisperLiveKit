## Triton Whisper Backend (Faster-Whisper over Triton)

This document describes how the optional Triton-backed Whisper ASR integrates with WhisperLiveKit.

### Overview

- Client-side backend class: `TritonWhisperASR` (`whisperlivekit/whisper_streaming_custom/backends.py`)
- Factory key: `--backend triton-whisper`
- Server: NVIDIA Triton Inference Server hosting Faster-Whisper (Python backend recommended)

### I/O Schema (Triton)

Model config example (`config.pbtxt`):

```
name: "faster_whisper"
backend: "python"
max_batch_size: 1

input [
  {
    name: "AUDIO_DATA"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]

output [
  {
    name: "RESULT"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

instance_group [
  {
    kind: KIND_GPU
    count: 1
    gpus: [ 0 ]
  }
]
```

Client sends:
- `AUDIO_DATA` (FP32, mono, 16 kHz, range [-1, 1], shape [num_samples])

Server returns in `RESULT` a JSON string with the following schema:

```json
{
  "segments": [
    {
      "start": 0.00,
      "end": 1.23,
      "no_speech_prob": 0.03,
      "words": [
        { "start": 0.10, "end": 0.32, "word": "hello", "probability": 0.97 },
        { "start": 0.33, "end": 0.60, "word": "world", "probability": 0.95 }
      ]
    }
  ],
  "info": { "language": "en", "duration": 1.23 }
}
```

Notes:
- Word-level timestamps are required for best streaming behavior.
- `segments[*].end` is used for segment-based trimming.

### Client Integration

- The `OnlineASRProcessor` feeds normalized audio to the selected backend.
- Selecting the Triton backend:
  - CLI/config: `--backend triton-whisper`
- `TritonWhisperASR` responsibilities:
  - Connect to Triton (gRPC recommended)
  - Send `AUDIO_DATA`
  - Parse `RESULT` JSON and expose:
    - `transcribe(audio, init_prompt) -> dict` (parsed JSON)
    - `ts_words(res) -> List[ASRToken]` (from `segments[*].words`)
    - `segments_end_ts(res) -> List[float]` (from `segments[*].end`)

### Assumptions & Requirements

- Input audio is float32 mono 16 kHz in range [-1, 1] (already handled upstream).
- If translation or VAD options are needed, extend the Triton model to accept control inputs or read request parameters.
- Maintain persistent Triton client connections to reduce overhead.

### Performance Tips

- Use gRPC client (`tritonclient.grpc`) for lower overhead and async support.
- Keep buffer trimming reasonable (e.g., 5–10 s) to limit payload size since the whole buffer is resent each step.
- Consider server-side session/context to avoid reprocessing long history.

### Troubleshooting

- Empty `RESULT` or malformed JSON: client will treat as no tokens.
- High latency: verify network path, batch size, and model compute type; ensure GPU placement is correct.


