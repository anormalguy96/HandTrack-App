# Hand Tracking Engine (Mobile)

Core production requirements:
- On-device inference
- High FPS with adaptive quality
- Minimal battery drain

## Plan
Implement as Flutter platform plugin:
- Android: MediaPipe Tasks + TFLite GPU delegate
- iOS: CoreML + Vision (or custom)

## Dart interface (expected)
- start(cameraConfig)
- stop()
- stream(landmarks, gestures, fps, latency)

## Efficiency must-haves
- ROI cropping
- frame skipping
- smoothing
- multi-thread pipeline
