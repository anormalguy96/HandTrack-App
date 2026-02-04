import Flutter
import AVFoundation
import Vision

final class VisionHandPoseEngine: NSObject, FlutterStreamHandler, AVCaptureVideoDataOutputSampleBufferDelegate {

  private var eventSink: FlutterEventSink?
  private let session = AVCaptureSession()
  private let output = AVCaptureVideoDataOutput()
  private var previewLayer: AVCaptureVideoPreviewLayer?

  private var preferFront: Bool
  private var running = true

  // simple FPS EMA
  private var lastTs: CFTimeInterval = 0
  private var fpsEma: Double = 0

  init(preferFront: Bool) {
    self.preferFront = preferFront
    super.init()
    configureSession()
  }

  func attachPreview(to view: UIView) {
    let layer = AVCaptureVideoPreviewLayer(session: session)
    layer.videoGravity = .resizeAspectFill
    layer.frame = view.bounds
    view.layer.addSublayer(layer)
    previewLayer = layer
  }

  func start() { session.startRunning() }
  func stop() { session.stopRunning() }

  func pause() { running = false }
  func resume() { running = true }

  func switchCamera() {
    preferFront.toggle()
    session.beginConfiguration()
    session.inputs.forEach { session.removeInput($0) }
    addCameraInput()
    session.commitConfiguration()
  }

  func applyConfig(_ cfg: [String: Any]) {
    // iOS Vision doesn’t use the same thresholds; kept for parity.
    if let front = cfg["preferFrontCamera"] as? Bool { preferFront = front }
  }

  func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
    eventSink = events
    return nil
  }

  func onCancel(withArguments arguments: Any?) -> FlutterError? {
    eventSink = nil
    return nil
  }

  private func configureSession() {
    session.beginConfiguration()
    session.sessionPreset = .high

    addCameraInput()

    output.alwaysDiscardsLateVideoFrames = true
    output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
    let queue = DispatchQueue(label: "vision_handpose_queue")
    output.setSampleBufferDelegate(self, queue: queue)

    if session.canAddOutput(output) { session.addOutput(output) }
    session.commitConfiguration()
  }

  private func addCameraInput() {
    let pos: AVCaptureDevice.Position = preferFront ? .front : .back
    guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: pos),
          let input = try? AVCaptureDeviceInput(device: device),
          session.canAddInput(input) else { return }
    session.addInput(input)
  }

  // Vision joints mapped to MediaPipe-like order (0 wrist, 1..4 thumb, 5..8 index, 9..12 middle, 13..16 ring, 17..20 little)
  private let orderedJoints: [VNHumanHandPoseObservation.JointName] = [
    .wrist,
    .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
    .indexMCP, .indexPIP, .indexDIP, .indexTip,
    .middleMCP, .middlePIP, .middleDIP, .middleTip,
    .ringMCP, .ringPIP, .ringDIP, .ringTip,
    .littleMCP, .littlePIP, .littleDIP, .littleTip
  ]

  func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
    guard running else { return }

    let now = CACurrentMediaTime()
    if lastTs != 0 {
      let inst = 1.0 / max(0.0001, (now - lastTs))
      fpsEma = fpsEma == 0 ? inst : (fpsEma * 0.9 + inst * 0.1)
    }
    lastTs = now

    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

    let request = VNDetectHumanHandPoseRequest()
    request.maximumHandCount = 1

    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
    do {
      try handler.perform([request])
      guard let obs = request.results?.first else {
        emitFrame(tsMs: Int(now * 1000), w: CVPixelBufferGetWidth(pixelBuffer), h: CVPixelBufferGetHeight(pixelBuffer), hands: [])
        return
      }

      let points = try obs.recognizedPoints(.all)
      var landmarks: [[String: Any]] = []
      landmarks.reserveCapacity(21)

      for j in orderedJoints {
        let p = points[j]
        // Vision provides normalized coordinates; y is bottom-left origin. Convert to top-left style: y = 1 - y
        let obj: [String: Any] = [
          "x": Double(p.x),
          "y": Double(1.0 - p.y),
          "z": 0.0,
          "conf": Double(p.confidence)
        ]
        landmarks.append(obj)
      }

      let hand: [String: Any] = [
        "handedness": "Unknown",
        "score": 1.0,
        "landmarks": landmarks
      ]

      emitFrame(
        tsMs: Int(now * 1000),
        w: CVPixelBufferGetWidth(pixelBuffer),
        h: CVPixelBufferGetHeight(pixelBuffer),
        hands: [hand]
      )
    } catch {
      // ignore occasional Vision errors
    }
  }

  private func emitFrame(tsMs: Int, w: Int, h: Int, hands: [[String: Any]]) {
    let payload: [String: Any] = [
      "tsMs": tsMs,
      "imageW": w,
      "imageH": h,
      "fps": fpsEma,
      "latencyMs": 0.0,
      "hands": hands
    ]
    if let data = try? JSONSerialization.data(withJSONObject: payload),
       let str = String(data: data, encoding: .utf8) {
      eventSink?(str)
    }
  }
}
