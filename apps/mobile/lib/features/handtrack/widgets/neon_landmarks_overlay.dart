import "dart:ui" as ui;
import "package:flutter/material.dart";
import "package:handtrack_engine/handtrack_engine.dart";

class NeonLandmarksOverlay extends StatelessWidget {
  const NeonLandmarksOverlay({super.key, required this.frame});
  final HandFrame? frame;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _Painter(frame),
    );
  }
}

class _Painter extends CustomPainter {
  _Painter(this.frame);
  final HandFrame? frame;

  @override
  void paint(Canvas canvas, Size size) {
    final f = frame;
    if (f == null || f.hands.isEmpty) return;

    final paint = Paint()
      ..style = PaintingStyle.fill
      ..color = const Color(0xFF69F0AE)
      ..maskFilter = const ui.MaskFilter.blur(ui.BlurStyle.normal, 10);

    for (final hand in f.hands) {
      for (final lm in hand.landmarks) {
        final x = lm.x * size.width;
        final y = lm.y * size.height;
        canvas.drawCircle(Offset(x, y), 5, paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant _Painter oldDelegate) => oldDelegate.frame != frame;
}
