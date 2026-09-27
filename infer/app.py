"""Local inference worker. Bind to loopback only. Stores nothing.

  uv run --python 3.12 uvicorn infer.app:app --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

import io
import os
import pickle
import tempfile
from pathlib import Path

import librosa
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "Model"

EMOTION_LEN = 259
DEPRESSION_LEN = 10293

app = FastAPI(title="MoodSense infer")
_models = None


def extract_mfcc(file_path: str, duration: float, sr: int, offset: float, n_mfcc: int):
    X, sample_rate = librosa.load(
        file_path, res_type="kaiser_fast", duration=duration, sr=sr, offset=offset
    )
    return np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=n_mfcc), axis=0)


def prepare_features(mfcc_features, required_length: int):
    current_length = len(mfcc_features)
    if current_length < required_length:
        padded = np.pad(
            mfcc_features, (0, required_length - current_length), "constant", constant_values=0
        )
    else:
        padded = mfcc_features[:required_length]
    return np.expand_dims(padded, axis=0)


def load_label_encoder(path: Path):
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        import joblib

        return joblib.load(path)


def get_models():
    global _models
    if _models is None:
        _models = {
            "emotion": load_model(MODEL_DIR / "emotion.keras"),
            "depression": load_model(MODEL_DIR / "depression.keras"),
            "lb_emo": load_label_encoder(MODEL_DIR / "lb-emotion.sav"),
            "lb_dp": load_label_encoder(MODEL_DIR / "lb-depression.sav"),
        }
    return _models


def predict_folder(folder: Path, duration: float, n_mfcc: int, required: int, model, encoder):
    labels: list[str] = []
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".wav", ".mp3", ".ogg", ".flac"})
    for path in files:
        try:
            mfccs = extract_mfcc(str(path), duration, 44100, 0.5, n_mfcc)
            x = prepare_features(mfccs, required)
            y_pred = model.predict(x, verbose=0)
            predicted_class = np.argmax(y_pred, axis=1)
            labels.append(str(encoder.inverse_transform(predicted_class)[0]))
        except Exception:
            continue
    return labels


def split_wav(audio: AudioSegment, chunk_ms: int, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for i, start in enumerate(range(0, len(audio), chunk_ms), start=1):
        chunk = audio[start : start + chunk_ms]
        chunk.export(dest / f"chunk_{i:04d}.wav", format="wav")


@app.on_event("startup")
def startup():
    missing = [
        MODEL_DIR / "emotion.keras",
        MODEL_DIR / "depression.keras",
        MODEL_DIR / "lb-emotion.sav",
        MODEL_DIR / "lb-depression.sav",
    ]
    for path in missing:
        if not path.is_file():
            raise RuntimeError(f"Missing model file: {path}")
    get_models()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
async def predict(request: Request):
    wav_bytes = await request.body()
    if len(wav_bytes) < 44:
        raise HTTPException(status_code=400, detail="WAV body required")

    models = get_models()
    try:
        audio = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode WAV: {exc}") from exc

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        split_wav(audio, 3 * 1000, tmp_path / "emotion")
        split_wav(audio, 2 * 60 * 1000, tmp_path / "depression")
        emotion = predict_folder(
            tmp_path / "emotion", 3, 13, EMOTION_LEN, models["emotion"], models["lb_emo"]
        )
        depression = predict_folder(
            tmp_path / "depression",
            2 * 60,
            20,
            DEPRESSION_LEN,
            models["depression"],
            models["lb_dp"],
        )

    return JSONResponse({"predictions": {"emotion": emotion, "depression": depression}})


def run():
    import uvicorn

    host = os.environ.get("INFERENCE_HOST", "127.0.0.1")
    port = int(os.environ.get("INFERENCE_PORT", "8001"))
    uvicorn.run("infer.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
