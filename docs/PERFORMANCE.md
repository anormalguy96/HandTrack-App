# Performance

## Targets
- 60 FPS on modern iPhone
- 30+ FPS on mid-tier Android using adaptive quality
- Smooth gesture response (p95 < 50ms)

## Adaptive quality levers
- input resolution
- ROI cropping
- frame skipping
- GPU delegate
- smoothing filters
- multi-thread pipeline (capture → inference → render)
