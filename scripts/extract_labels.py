"""Read Model/lb-emotion.sav and Model/lb-depression.sav into src/lib/inference/labels.json.

Label order is sklearn LabelEncoder.classes_ (model index order). Depression must
contain 'med' and must not contain 'mid'.
"""

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "Model"
OUT = ROOT / "src" / "lib" / "inference" / "labels.json"


def load_classes(path: Path) -> list[str]:
    try:
        with path.open("rb") as handle:
            encoder = pickle.load(handle)
    except Exception:
        import joblib

        encoder = joblib.load(path)
    classes = getattr(encoder, "classes_", None)
    if classes is None:
        raise SystemExit(f"{path} has no classes_ attribute")
    return [str(label) for label in classes]


def main() -> None:
    emotion_path = MODEL_DIR / "lb-emotion.sav"
    depression_path = MODEL_DIR / "lb-depression.sav"
    for path in (emotion_path, depression_path):
        if not path.is_file():
            raise SystemExit(f"Missing label encoder: {path}")

    emotion = load_classes(emotion_path)
    depression = load_classes(depression_path)
    lowered = [label.lower() for label in depression]
    if "mid" in lowered:
        raise SystemExit("depression labels contain 'mid'; expected 'med'")
    if "med" not in lowered:
        raise SystemExit("depression labels are missing 'med'")

    payload = {"emotion": emotion, "depression": depression}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"  emotion ({len(emotion)}): {emotion}")
    print(f"  depression ({len(depression)}): {depression}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
