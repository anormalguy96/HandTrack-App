import Flutter
import UIKit

public class HandtrackEnginePlugin: NSObject, FlutterPlugin {
  public static func register(with registrar: FlutterPluginRegistrar) {
    let factory = HandtrackViewFactory(messenger: registrar.messenger())
    registrar.register(factory, withId: "handtrack_engine/view")
  }
}

final class HandtrackViewFactory: NSObject, FlutterPlatformViewFactory {
  private let messenger: FlutterBinaryMessenger
  init(messenger: FlutterBinaryMessenger) { self.messenger = messenger }

  func createArgsCodec() -> FlutterMessageCodec & NSObjectProtocol { FlutterStandardMessageCodec.sharedInstance() }

  func create(
    withFrame frame: CGRect,
    viewIdentifier viewId: Int64,
    arguments args: Any?
  ) -> FlutterPlatformView {
    let params = (args as? [String: Any]) ?? [:]
    return HandtrackPlatformView(frame: frame, viewId: viewId, messenger: messenger, params: params)
  }
}
