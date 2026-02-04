library handtrack_engine;

import "dart:async";
import "dart:convert";

import "package:flutter/foundation.dart";
import "package:flutter/services.dart";
import "package:flutter/widgets.dart";

part "src/models.dart";

/// PlatformView widget that renders native camera preview and streams hand frames.
class HandtrackView extends StatefulWidget {
  const HandtrackView({
    super.key,
    required this.onFrame,
    this.config = const HandtrackConfig(),
    this.onError,
  });

  final HandtrackConfig config;
  final ValueChanged<HandFrame> onFrame;
  final ValueChanged<String>? onError;

  @override
  State<HandtrackView> createState() => _HandtrackViewState();
}

class _HandtrackViewState extends State<HandtrackView> {
  static const _viewType = "handtrack_engine/view";
  int? _viewId;
  StreamSubscription? _sub;

  @override
  void dispose() {
    _sub?.cancel();
    super.dispose();
  }

  void _bindStream(int id) {
    _sub?.cancel();
    final events = EventChannel("handtrack_engine/events_$id");
    _sub = events.receiveBroadcastStream().listen((evt) {
      try {
        final map = (evt is String) ? jsonDecode(evt) as Map<String, dynamic> : (evt as Map).cast<String, dynamic>();
        widget.onFrame(HandFrame.fromJson(map));
      } catch (e) {
        widget.onError?.call("Parse error: $e");
      }
    }, onError: (e) {
      widget.onError?.call("Engine error: $e");
    });
  }

  @override
  Widget build(BuildContext context) {
    final creationParams = widget.config.toJson();

    if (defaultTargetPlatform == TargetPlatform.android) {
      return AndroidView(
        viewType: _viewType,
        creationParams: creationParams,
        creationParamsCodec: const StandardMessageCodec(),
        onPlatformViewCreated: (id) {
          _viewId = id;
          _bindStream(id);
        },
      );
    }

    if (defaultTargetPlatform == TargetPlatform.iOS) {
      return UiKitView(
        viewType: _viewType,
        creationParams: creationParams,
        creationParamsCodec: const StandardMessageCodec(),
        onPlatformViewCreated: (id) {
          _viewId = id;
          _bindStream(id);
        },
      );
    }

    return const SizedBox();
  }
}

/// Per-view control commands (optional, used for quality adaptation).
class HandtrackController {
  HandtrackController._(this._viewId);

  final int _viewId;
  MethodChannel get _ch => MethodChannel("handtrack_engine/methods_$_viewId");

  static HandtrackController forViewId(int viewId) => HandtrackController._(viewId);

  Future<void> setConfig(HandtrackConfig cfg) => _ch.invokeMethod("setConfig", cfg.toJson());

  Future<void> pause() => _ch.invokeMethod("pause");

  Future<void> resume() => _ch.invokeMethod("resume");

  Future<void> switchCamera() => _ch.invokeMethod("switchCamera");
}
