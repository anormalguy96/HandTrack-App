import "dart:ui" as ui;
import "package:flutter/material.dart";
import "package:handtrack_engine/handtrack_engine.dart";

class NeonLandmarksOverlay extends StatelessWidget {
  const NeonLandmarksOverlay({super.key, required this.frame});
  final HandFrame? frame;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _SkeletonPainter(frame),
    );
  }
}

class _SkeletonPainter extends CustomPainter {
  _SkeletonPainter(this.frame);
  final HandFrame? frame;

  static const _connections = [
    [0, 1], [1, 2], [2, 3], [3, 4], // thumb
    [0, 5], [5, 6], [6, 7], [7, 8], // index
    [0, 9], [9, 10], [10, 11], [11, 12], // middle
    [0, 13], [13, 14], [14, 15], [15, 16], // ring
    [0, 17], [17, 18], [18, 19], [19, 20], // pinky
    [5, 9], [9, 13], [13, 17], // palm base
  ];

  @override
  void paint(Canvas canvas, Size size) {
    final f = frame;
    if (f == null || f.hands.isEmpty) return;

    final dotPaint = Paint()
      ..color = const Color(0xFF69F0AE)
      ..style = PaintingStyle.fill;

    final linePaint = Paint()
      ..color = const Color(0xFF69F0AE).withOpacity(0.5)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    final glowPaint = Paint()
      ..color = const Color(0xFF69F0AE).withOpacity(0.3)
      ..strokeWidth = 4.0
      ..style = PaintingStyle.stroke
      ..maskFilter = const ui.MaskFilter.blur(ui.BlurStyle.normal, 4);

    for (final hand in f.hands) {
      final points = hand.landmarks
          .map((lm) => Offset(lm.x * size.width, lm.y * size.height))
          .toList();

      // 1. Draw connections
      for (final conn in _connections) {
        if (conn[0] < points.length && conn[1] < points.length) {
          canvas.drawPath(
            Path()
              ..moveTo(points[conn[0]].dx, points[conn[0]].dy)
              ..lineTo(points[conn[1]].dx, points[conn[1]].dy),
            glowPaint,
          );
          canvas.drawLine(points[conn[0]], points[conn[1]], linePaint);
        }
      }

      // 2. Draw dots
      for (final p in points) {
        canvas.drawCircle(p, 4, dotPaint);
      }

      // Index tip glow
      canvas.drawCircle(
          points[8],
          8,
          Paint()
            ..color = const Color(0xFF69F0AE).withOpacity(0.2)
            ..maskFilter = const ui.MaskFilter.blur(ui.BlurStyle.normal, 6));
    }
  }

  @override
  bool shouldRepaint(covariant _SkeletonPainter oldDelegate) =>
      oldDelegate.frame != frame;
}
