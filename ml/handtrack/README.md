# ML (R&D)

This folder is for:
- benchmarking capture pipelines
- experimenting with gesture classifiers (optional)
- evaluation / dataset tooling

Production inference for mobile is done on-device:
- Android: MediaPipe Tasks Vision HandLandmarker (model bundle)
- iOS: Vision hand pose detection

If you later add a custom gesture classifier:
- take the 21x3 landmarks stream
- train a lightweight model
- export to TFLite (Android) and CoreML (iOS)
