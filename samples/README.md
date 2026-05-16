# Sample Audio Files

Place sample audio files here for manual testing and demos.

## Requirements

- **Format**: WAV, FLAC, OGG, or any format supported by libsndfile
- **Sample rate**: Any (will be resampled to 16 kHz automatically)
- **Channels**: Mono preferred (stereo will be mixed down)
- **Duration**: At least 1 second for reliable analysis

## Testing with curl

```bash
# POST endpoint
curl -X POST http://localhost:8000/analyze \
  -F "file=@samples/your_audio.wav"

# WebSocket (using websocat)
websocat ws://localhost:8000/ws/analyze < samples/your_audio.raw
```

## Generating Test Audio

You can generate a synthetic test tone with Python:

```python
import numpy as np
import soundfile as sf

sr = 16000
t = np.linspace(0, 3, sr * 3, dtype=np.float32)
signal = 0.5 * np.sin(2 * np.pi * 440 * t)
sf.write("samples/test_tone.wav", signal, sr)
```
