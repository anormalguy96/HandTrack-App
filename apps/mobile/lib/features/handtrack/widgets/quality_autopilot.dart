import "package:flutter/material.dart";
import "package:handtrack_engine/handtrack_engine.dart";

class QualityAutopilot extends StatefulWidget {
  const QualityAutopilot({
    super.key,
    required this.frame,
    required this.current,
    required this.onApply,
  });

  final HandFrame? frame;
  final HandtrackConfig current;
  final ValueChanged<HandtrackConfig> onApply;

  @override
  State<QualityAutopilot> createState() => _QualityAutopilotState();
}

class _QualityAutopilotState extends State<QualityAutopilot> {
  int _lowStreak = 0;
  int _highStreak = 0;

  @override
  Widget build(BuildContext context) {
    final fps = widget.frame?.fps ?? 0;

    // simple logic: keep it stable; adjust slowly
    if (fps > 50) {
      _highStreak++;
      _lowStreak = 0;
    } else if (fps < 24) {
      _lowStreak++;
      _highStreak = 0;
    } else {
      _lowStreak = 0;
      _highStreak = 0;
    }

    if (_lowStreak >= 30 && widget.current.targetFps > 24) {
      _lowStreak = 0;
      widget.onApply(_copy(widget.current, targetFps: 24));
    }

    if (_highStreak >= 60 && widget.current.targetFps < 60) {
      _highStreak = 0;
      widget.onApply(_copy(widget.current, targetFps: 60));
    }

    return const SizedBox(height: 1); // invisible; pure controller
  }

  HandtrackConfig _copy(HandtrackConfig c, {required int targetFps}) => HandtrackConfig(
        maxHands: c.maxHands,
        minDetection: c.minDetection,
        minPresence: c.minPresence,
        minTracking: c.minTracking,
        targetFps: targetFps,
        preferFrontCamera: c.preferFrontCamera,
        enablePreview: c.enablePreview,
        androidDelegate: c.androidDelegate,
      );
}
