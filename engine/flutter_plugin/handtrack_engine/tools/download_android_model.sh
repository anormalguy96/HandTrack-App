#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ASSETS="$ROOT/android/src/main/assets"
mkdir -p "$ASSETS"

MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
OUT="$ASSETS/hand_landmarker.task"

echo "Downloading model -> $OUT"
curl -L "$MODEL_URL" -o "$OUT"
echo "Done."
