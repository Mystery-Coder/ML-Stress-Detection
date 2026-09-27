"""Convert Model/emotion.keras and Model/depression.keras to TFJS layers models.

Requires tensorflowjs:
  pip install tensorflowjs
  python scripts/convert_keras_to_tfjs.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = (
    (ROOT / "Model" / "emotion.keras", ROOT / "models" / "tfjs_model" / "emotion"),
    (ROOT / "Model" / "depression.keras", ROOT / "models" / "tfjs_model" / "depression"),
)


def converter_command() -> list[str]:
    found = shutil.which("tensorflowjs_converter")
    if found:
        return [found]
    return [sys.executable, "-m", "tensorflowjs.converters.converter"]


def main() -> None:
    base = converter_command()
    for src, dst in PAIRS:
        if not src.is_file():
            raise SystemExit(f"Missing Keras model: {src}")
        dst.mkdir(parents=True, exist_ok=True)
        cmd = [
            *base,
            "--input_format=keras",
            "--output_format=tfjs_layers_model",
            str(src),
            str(dst),
        ]
        print(" ".join(cmd))
        subprocess.check_call(cmd)
        print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
