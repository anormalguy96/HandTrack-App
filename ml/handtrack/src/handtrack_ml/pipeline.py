"""Starter R&D pipeline.

Goal:
- prototype hand landmarks extraction
- evaluate FPS vs accuracy tradeoffs
- export mobile-ready artifacts (TFLite/CoreML)

This file intentionally avoids a full training stack until requirements are locked.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import mediapipe as mp


@dataclass
class RunStats:
    frames: int
    seconds: float

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0


def run_webcam_preview(max_seconds: int = 10) -> RunStats:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No webcam found.")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    start = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _ = hands.process(rgb)

        frames += 1
        if time.time() - start >= max_seconds:
            break

    cap.release()
    hands.close()
    return RunStats(frames=frames, seconds=time.time() - start)


if __name__ == "__main__":
    stats = run_webcam_preview()
    print(f"Frames: {stats.frames}, Seconds: {stats.seconds:.2f}, FPS: {stats.fps:.1f}")
