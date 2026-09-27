"""Librosa reference for scripts/parity_test.mjs.

Builds a deterministic 440 Hz tone and writes mean-over-coefficient MFCCs
(np.mean(librosa.feature.mfcc(...), axis=0)) to scripts/fixtures/mfcc_reference.json.

Params match training extract_mfcc for emotion: sr=44100, offset=0.5, duration=3,
n_mfcc=13, n_fft=2048, hop_length=512, n_mels=128.
"""

import json
import math
from pathlib import Path

import numpy as np

try:
    import librosa
except ImportError as exc:
    raise SystemExit("librosa is required: pip install librosa") from exc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scripts" / "fixtures" / "mfcc_reference.json"

SR = 44100
SECONDS = 3.5
OFFSET = 0.5
DURATION = 3.0
N_MFCC = 13
N_FFT = 2048
HOP = 512
N_MELS = 128
FREQ = 440.0
AMPLITUDE = 0.2


def tone() -> np.ndarray:
    count = int(SR * SECONDS)
    samples = np.empty(count, dtype=np.float32)
    for i in range(count):
        samples[i] = AMPLITUDE * math.sin(2 * math.pi * FREQ * i / SR)
    return samples


def main() -> None:
    y = tone()
    start = int(OFFSET * SR)
    end = start + int(DURATION * SR)
    clip = y[start:end]
    mfcc = librosa.feature.mfcc(
        y=clip,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
    )
    mean = np.mean(mfcc, axis=0)
    payload = {
        "sr": SR,
        "seconds": SECONDS,
        "offset": OFFSET,
        "duration": DURATION,
        "n_mfcc": N_MFCC,
        "n_fft": N_FFT,
        "hop_length": HOP,
        "n_mels": N_MELS,
        "frequency_hz": FREQ,
        "amplitude": AMPLITUDE,
        "n_frames": int(mean.shape[0]),
        "mean_mfcc": [float(value) for value in mean],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"n_frames={payload['n_frames']} (expected length of mean axis=0)")
    print("Next: node --experimental-strip-types scripts/parity_test.mjs")


if __name__ == "__main__":
    main()
