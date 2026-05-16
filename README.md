# Voice Attribute Service

A **streaming-first FastAPI backend** for real-time voice attribute analysis. Extracts speaker embeddings using SpeechBrain's ECAPA-TDNN model and classifies gender, age group, audio quality, and spoken language.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     FastAPI Application                       │
│                                                              │
│  ┌─────────────────┐         ┌─────────────────────────┐    │
│  │  WebSocket       │         │  POST /analyze           │    │
│  │  /ws/analyze     │         │  (compatibility)         │    │
│  │  (primary)       │         │                          │    │
│  └────────┬────────┘         └──────────┬──────────────┘    │
│           │                              │                    │
│           ▼                              ▼                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              audio.py — resample to 16kHz mono       │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │   inference.py — SpeechBrain ECAPA-TDNN embedding    │    │
│  │                  (192-dim vector)                     │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                          │                                    │
│           ┌──────────────┼──────────────┐                    │
│           ▼              ▼              ▼                    │
│  ┌──────────────┐ ┌───────────┐ ┌──────────────┐           │
│  │ classifier.py│ │ quality.py│ │ language.py  │           │
│  │ gender / age │ │ SNR/clip  │ │ VoxLingua107 │           │
│  └──────────────┘ └───────────┘ └──────────────┘           │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package manager)

### Install & Run

```bash
# Clone the repository
git clone <repo-url>
cd voice-attribute-service

# Install dependencies
uv sync

# Run the development server
uv run uvicorn app.main:app --reload

# Open API docs
open http://localhost:8000/docs
```

### Docker

```bash
# Build and run
docker compose up --build

# Or build manually
docker build -t voice-attribute-service .
docker run -p 8000:8000 voice-attribute-service
```

## API Endpoints

### WebSocket `/ws/analyze` (Primary)

Real-time streaming voice analysis. Connect with a WebSocket client and stream binary audio data (PCM float32, 16kHz).

**Protocol:**
1. Connect to `ws://localhost:8000/ws/analyze`
2. *(Optional)* Send JSON config: `{"sample_rate": 16000, "encoding": "pcm_f32le"}`
3. Send binary audio chunks
4. Receive JSON `StreamingChunk` results per processed chunk
5. Send `"END"` text message to finalize
6. Receive final aggregated `AnalysisResult`

**Example with Python:**
```python
import asyncio
import websockets
import struct
import numpy as np

async def stream_audio():
    async with websockets.connect("ws://localhost:8000/ws/analyze") as ws:
        # Generate and send audio
        sr = 16000
        audio = np.random.randn(sr * 5).astype(np.float32) * 0.1
        chunk_size = sr * 2  # 2-second chunks

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            await ws.send(struct.pack(f"<{len(chunk)}f", *chunk))

            # Check for responses
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=0.1)
                print(f"Chunk result: {response}")
            except asyncio.TimeoutError:
                pass

        # Signal end of stream
        await ws.send("END")
        final = await ws.recv()
        print(f"Final result: {final}")

asyncio.run(stream_audio())
```

### POST `/analyze` (Compatibility)

Upload a complete audio file for analysis.

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@samples/audio.wav"
```

**Response:**
```json
{
  "gender": "female",
  "gender_confidence": 0.9234,
  "age_group": "adult",
  "age_confidence": 0.7891,
  "language": "en",
  "language_confidence": 0.87,
  "quality": {
    "snr_db": 25.4,
    "clipping_ratio": 0.0,
    "duration_seconds": 3.5,
    "is_acceptable": true
  }
}
```

### GET `/health`

Health check with model status.

```bash
curl http://localhost:8000/health
```

## Training Pipeline

The training scripts prepare data, extract embeddings, and train classifiers.

### Full Pipeline

```bash
# Run everything in sequence
uv run python scripts/train_all.py
```

### Step by Step

```bash
# 1. Prepare gender manifest from VoxCeleb metadata
uv run python scripts/prepare_voxceleb_gender.py \
    --meta data/raw/vox1_meta.csv \
    --audio-dir data/raw/voxceleb1/wav

# 2. Prepare age manifest from AgeVoxCeleb metadata
uv run python scripts/prepare_agevoxceleb.py \
    --meta data/raw/agevoxceleb_meta.csv \
    --audio-dir data/raw/voxceleb1/wav

# 3. Extract ECAPA-TDNN embeddings
uv run python scripts/extract_embeddings.py \
    --manifest data/processed/gender_manifest.csv \
    --label-col gender \
    --output data/embeddings/gender_embeddings.npz

uv run python scripts/extract_embeddings.py \
    --manifest data/processed/age_manifest.csv \
    --label-col age_group \
    --output data/embeddings/age_embeddings.npz

# 4. Train classifiers
uv run python scripts/train_gender_classifier.py
uv run python scripts/train_age_classifier.py

# 5. Evaluate on Common Voice (optional)
uv run python scripts/eval_common_voice.py \
    --tsv data/raw/common_voice/validated.tsv \
    --clips-dir data/raw/common_voice/clips
```

### Data Requirements

| Dataset      | Purpose          | Source                                           |
|-------------|------------------|--------------------------------------------------|
| VoxCeleb1/2 | Gender labels    | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/  |
| AgeVoxCeleb | Age labels       | https://github.com/DigitalPhonetics/AgeVoxCeleb  |
| Common Voice| Evaluation only  | https://commonvoice.mozilla.org/                  |

## Project Structure

```
voice-attribute-service/
├── app/                          # FastAPI application
│   ├── main.py                   # Entry point, lifespan, CORS
│   ├── api.py                    # REST routes (POST /analyze, GET /health)
│   ├── streaming.py              # WebSocket handler (/ws/analyze)
│   ├── audio.py                  # Audio decoding, resampling, chunking
│   ├── inference.py              # SpeechBrain ECAPA-TDNN embedding extraction
│   ├── classifier.py             # Gender/age classifiers (joblib)
│   ├── quality.py                # Audio quality metrics (SNR, clipping)
│   ├── language.py               # Language detection (SpeechBrain VoxLingua107)
│   ├── schemas.py                # Pydantic request/response models
│   ├── config.py                 # Module-level configuration constants
│   └── logger.py                 # Structured JSON logging
├── scripts/                      # Training pipeline scripts
│   ├── prepare_voxceleb_gender.py
│   ├── prepare_agevoxceleb.py
│   ├── extract_embeddings.py
│   ├── train_gender_classifier.py
│   ├── train_age_classifier.py
│   ├── train_all.py
│   └── eval_common_voice.py
├── data/                         # Training data (git-ignored)
│   ├── raw/                      # Downloaded datasets
│   ├── processed/                # Cleaned manifests
│   └── embeddings/               # Extracted .npz embedding files
├── models/                       # Trained classifiers (git-ignored)
├── tests/                        # pytest test suite
├── samples/                      # Sample audio for manual testing
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml                # uv project configuration
└── requirements.txt              # Pinned deps for Docker
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_health.py -v
```

## Configuration

| Variable                      | Default                                  | Description                      |
|-------------------------------|------------------------------------------|----------------------------------|
| `SAMPLE_RATE`                 | `16000`                                  | Target audio sample rate         |
| `STREAM_MIN_SECONDS`          | `3.0`                                    | Minimum audio length (seconds)   |
| `STREAM_INFERENCE_INTERVAL_SECONDS` | `1.0`                              | Streaming inference interval     |
| `SPEECHBRAIN_SPEAKER_MODEL`   | `speechbrain/spkrec-ecapa-voxceleb`      | Speaker embedding model          |
| `SPEECHBRAIN_LANGUAGE_MODEL`  | `speechbrain/lang-id-voxlingua107-ecapa` | Language-ID model                |
| `ENABLE_LANGUAGE_DETECTION`   | `true`                                   | Toggle language detection        |
| `LANGUAGE_CONFIDENCE_THRESHOLD` | `0.50`                                 | Min confidence for language ID   |
| `MOCK_MODEL`                  | `false`                                  | Use mock models for testing      |
| `GENDER_CONFIDENCE_THRESHOLD` | `0.60`                                   | Min confidence for gender        |
| `AGE_CONFIDENCE_THRESHOLD`    | `0.45`                                   | Min confidence for age group     |

## Language Detection

Language detection is implemented using SpeechBrain's pretrained **VoxLingua107 ECAPA** model (`speechbrain/lang-id-voxlingua107-ecapa`). No custom language classifier is trained. The model supports broad multilingual language identification (107 languages) and is downloaded/cached automatically by SpeechBrain on first run.

Language detection runs directly on the same preprocessed 3–5 second audio window used by the streaming pipeline. It is **separate** from the gender/age pipeline:
- Gender/age uses speaker embeddings + trained scikit-learn classifiers.
- Language detection directly uses the audio waveform via SpeechBrain's `classify_batch` method.

No language dataset is required for implementation. Language datasets are only needed if you want to run a separate language evaluation harness.

## License

MIT
