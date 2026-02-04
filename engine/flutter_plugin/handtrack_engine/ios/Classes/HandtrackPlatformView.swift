import Flutter
import UIKit

final class HandtrackPlatformView: NSObject, FlutterPlatformView {
  private let container: UIView
  private let engine: VisionHandPoseEngine
  private let eventChannel: FlutterEventChannel
  private let methodChannel: FlutterMethodChannel

  init(frame: CGRect, viewId: Int64, messenger: FlutterBinaryMessenger, params: [String: Any]) {
    container = UIView(frame: frame)
    container.backgroundColor = .black

    eventChannel = FlutterEventChannel(name: "handtrack_engine/events_\(viewId)", binaryMessenger: messenger)
    methodChannel = FlutterMethodChannel(name: "handtrack_engine/methods_\(viewId)", binaryMessenger: messenger)

    engine = VisionHandPoseEngine(preferFront: (params["preferFrontCamera"] as? Bool) ?? true)

    super.init()

    eventChannel.setStreamHandler(engine)
    engine.attachPreview(to: container)

    methodChannel.setMethodCallHandler { [weak self] call, result in
      guard let self = self else { return }
      switch call.method {
      case "pause":
        self.engine.pause()
        result(nil)
      case "resume":
        self.engine.resume()
        result(nil)
      case "switchCamera":
        self.engine.switchCamera()
        result(nil)
      case "setConfig":
        if let m = call.arguments as? [String: Any] {
          self.engine.applyConfig(m)
        }
        result(nil)
      default:
        result(FlutterMethodNotImplemented)
      }
    }

    engine.start()
  }

  func view() -> UIView { container }

  deinit { engine.stop() }
}
