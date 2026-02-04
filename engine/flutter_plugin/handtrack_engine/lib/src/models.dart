part of handtrack_engine;

@immutable
class HandtrackConfig {
  const HandtrackConfig({
    this.maxHands = 1,
    this.minDetection = 0.5,
    this.minPresence = 0.5,
    this.minTracking = 0.5,
    this.targetFps = 30,
    this.preferFrontCamera = true,
    this.enablePreview = true,
    this.androidDelegate = "CPU", // CPU | GPU (GPU optional depending on device)
  });

  final int maxHands;
  final double minDetection;
  final double minPresence;
  final double minTracking;
  final int targetFps;
  final bool preferFrontCamera;
  final bool enablePreview;
  final String androidDelegate;

  Map<String, dynamic> toJson() => {
        "maxHands": maxHands,
        "minDetection": minDetection,
        "minPresence": minPresence,
        "minTracking": minTracking,
        "targetFps": targetFps,
        "preferFrontCamera": preferFrontCamera,
        "enablePreview": enablePreview,
        "androidDelegate": androidDelegate,
      };
}

@immutable
class HandFrame {
  const HandFrame({
    required this.tsMs,
    required this.imageW,
    required this.imageH,
    required this.fps,
    required this.latencyMs,
    required this.hands,
  });

  final int tsMs;
  final int imageW;
  final int imageH;
  final double fps;
  final double latencyMs;
  final List<Hand> hands;

  static HandFrame fromJson(Map<String, dynamic> json) => HandFrame(
        tsMs: (json["tsMs"] as num).toInt(),
        imageW: (json["imageW"] as num).toInt(),
        imageH: (json["imageH"] as num).toInt(),
        fps: (json["fps"] as num).toDouble(),
        latencyMs: (json["latencyMs"] as num).toDouble(),
        hands: (json["hands"] as List)
            .map((h) => Hand.fromJson((h as Map).cast<String, dynamic>()))
            .toList(growable: false),
      );
}

@immutable
class Hand {
  const Hand({
    required this.handedness,
    required this.score,
    required this.landmarks,
  });

  final String handedness; // "Left" | "Right" | "Unknown"
  final double score;
  final List<Landmark> landmarks; // 21

  static Hand fromJson(Map<String, dynamic> json) => Hand(
        handedness: (json["handedness"] as String?) ?? "Unknown",
        score: (json["score"] as num?)?.toDouble() ?? 0,
        landmarks: (json["landmarks"] as List)
            .map((p) => Landmark.fromJson((p as Map).cast<String, dynamic>()))
            .toList(growable: false),
      );
}

@immutable
class Landmark {
  const Landmark({required this.x, required this.y, required this.z, required this.conf});

  /// normalized [0..1]
  final double x;
  final double y;

  /// normalized-ish (platform dependent)
  final double z;

  final double conf;

  static Landmark fromJson(Map<String, dynamic> json) => Landmark(
        x: (json["x"] as num).toDouble(),
        y: (json["y"] as num).toDouble(),
        z: (json["z"] as num?)?.toDouble() ?? 0,
        conf: (json["conf"] as num?)?.toDouble() ?? 1,
      );
}
