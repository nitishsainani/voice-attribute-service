"""
Manual WebSocket test script.

Loads a real audio file, converts to PCM float32, and streams it
to ws://localhost:8000/ws/analyze in chunks.

Usage:
    uv run python scripts/test_ws_manually.py
    uv run python scripts/test_ws_manually.py --file path/to/audio.mp3
"""

import argparse
import asyncio
import json
import struct
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets


async def stream_audio(file_path: str) -> None:
    print(f"\n📂 Loading: {file_path}")
    data, sr = sf.read(file_path, dtype="float32")

    # Mix to mono if stereo
    if data.ndim == 2:
        data = data.mean(axis=1)

    print(f"   Sample rate: {sr} Hz | Duration: {len(data)/sr:.2f}s | Samples: {len(data)}")

    uri = "ws://localhost:8000/ws/analyze"
    print(f"\n🔗 Connecting to {uri}...")

    async with websockets.connect(uri) as ws:
        # Send config
        config = {"sample_rate": sr, "encoding": "pcm_f32le"}
        await ws.send(json.dumps(config))
        ack = await ws.recv()
        print(f"✅ Config ack: {ack}\n")

        # Stream audio in 1-second chunks
        chunk_size = sr  # 1 second of audio
        chunk_count = 0

        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]
            pcm_bytes = struct.pack(f"<{len(chunk)}f", *chunk)
            await ws.send(pcm_bytes)
            chunk_count += 1

            # Check for responses (non-blocking)
            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    parsed = json.loads(response)
                    if parsed.get("type") == "chunk":
                        print(
                            f"   🎤 Chunk {parsed['chunk_index']}: "
                            f"gender={parsed['gender']} ({parsed['gender_confidence']:.2f}) | "
                            f"age={parsed['age_group']} ({parsed['age_confidence']:.2f}) | "
                            f"lang={parsed['language']} ({parsed['language_confidence']:.2f})"
                        )
            except asyncio.TimeoutError:
                pass

        # Signal end
        print(f"\n📤 Sent {chunk_count} chunks. Sending END...")
        await ws.send("END")

        # Collect remaining responses
        try:
            while True:
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                parsed = json.loads(response)

                if parsed.get("type") == "chunk":
                    print(
                        f"   🎤 Chunk {parsed['chunk_index']}: "
                        f"gender={parsed['gender']} ({parsed['gender_confidence']:.2f}) | "
                        f"age={parsed['age_group']} ({parsed['age_confidence']:.2f}) | "
                        f"lang={parsed['language']} ({parsed['language_confidence']:.2f})"
                    )
                elif parsed.get("type") == "final":
                    result = parsed["result"]
                    print(f"\n{'='*60}")
                    print(f"🏁 FINAL RESULT")
                    print(f"{'='*60}")
                    print(f"   Gender:   {result['gender']} ({result['gender_confidence']:.4f})")
                    print(f"   Age:      {result['age_group']} ({result['age_confidence']:.4f})")
                    print(f"   Language: {result['language']} ({result['language_confidence']:.4f})")
                    q = result["quality"]
                    print(f"   SNR:      {q['snr_db']:.1f} dB")
                    print(f"   Duration: {q['duration_seconds']:.2f}s")
                    print(f"   Quality:  {'✅ acceptable' if q['is_acceptable'] else '❌ poor'}")
                    print(f"{'='*60}\n")
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
            pass


def main():
    parser = argparse.ArgumentParser(description="Manual WebSocket test")
    parser.add_argument(
        "--file",
        type=str,
        default="data/raw/common_voice_spontaneous_en/audios/spontaneous-speech-en-1.mp3",
        help="Audio file to stream",
    )
    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"❌ File not found: {args.file}")
        return

    asyncio.run(stream_audio(args.file))


if __name__ == "__main__":
    main()
