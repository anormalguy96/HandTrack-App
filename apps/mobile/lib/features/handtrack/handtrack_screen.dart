import "dart:async";
import "dart:ui" as ui;
import "dart:io";

import "package:flutter/material.dart";
import "package:handtrack_engine/handtrack_engine.dart";
import "package:screenshot/screenshot.dart";
import "package:share_plus/share_plus.dart";
import "package:path_provider/path_provider.dart";

import "logic/gesture_processor.dart";
import "logic/drawing_logic.dart";
import "logic/voice_logic.dart";
import "widgets/neon_landmarks_overlay.dart";

class HandtrackScreen extends StatefulWidget {
  const HandtrackScreen({super.key});

  @override
  State<HandtrackScreen> createState() => _HandtrackScreenState();
}

class _HandtrackScreenState extends State<HandtrackScreen> {
  final DrawingController _drawController = DrawingController();
  final ScreenshotController _screenshotController = ScreenshotController();
  final VoiceController _voiceController = VoiceController();

  HandFrame? _lastFrame;
  HandtrackConfig _cfg =
      const HandtrackConfig(targetFps: 24, preferFrontCamera: true);

  // Interaction State
  DrawingStroke? _activeStroke;
  DrawingStroke? _grabbedStroke;
  Offset? _lastGrabPos;
  String _status = "Initializing...";
  String _gestureName = "None";

  @override
  void initState() {
    super.initState();
    _drawController.addListener(() => setState(() {}));
    _initVoice();
  }

  Future<void> _initVoice() async {
    await _voiceController.init();
    _voiceController.onCommand = _handleVoiceCommand;
    _voiceController.onStatus = (s) => setState(() => _status = s);
    _voiceController.speak("Neon Studio mobile active.");
  }

  void _handleVoiceCommand(String text) {
    if (!mounted) return;
    final t = text.toLowerCase();
    if (t.contains("undo")) _drawController.undo();
    if (t.contains("redo")) _drawController.redo();
    if (t.contains("clear")) _drawController.clear();
    if (t.contains("share") || t.contains("export")) _share();

    if (t.contains("reset")) _resetEngine();

    setState(() => _status = "Heard: \"$text\"");
  }

  void _resetEngine() async {
    await _voiceController.stopListening();
    _drawController.clear();
    _initVoice();
    setState(() => _status = "System Reset.");
  }

  void _onFrame(HandFrame frame) {
    if (!mounted) return;
    setState(() {
      _lastFrame = frame;
      _status = "Ready";
    });

    if (frame.hands.isEmpty) {
      _finishStroke();
      _grabbedStroke = null;
      return;
    }

    final hand = frame.hands.first;
    final gesture = GestureProcessor.process(hand);
    _gestureName = GestureProcessor.gestureName(gesture);

    final indexTip = Offset(hand.landmarks[8].x, hand.landmarks[8].y);

    if (gesture == HandGesture.pointingUp) {
      _handlePointing(indexTip);
    } else if (gesture == HandGesture.closedFist ||
        gesture == HandGesture.thumbUp) {
      _handleGrabbing(indexTip);
    } else {
      _finishStroke();
      _grabbedStroke = null;
    }
  }

  void _handlePointing(Offset tip) {
    if (_activeStroke == null) {
      _activeStroke = DrawingStroke(
        points: [DrawingPoint(x: tip.dx, y: tip.dy)],
        color: const Color(0xFF69F0AE),
        thickness: 6.0,
      );
      _drawController.addStroke(_activeStroke!);
    } else {
      _activeStroke!.points.add(DrawingPoint(x: tip.dx, y: tip.dy));
      _drawController.updateLastStroke(_activeStroke!);
    }
    _grabbedStroke = null;
  }

  void _handleGrabbing(Offset tip) {
    _finishStroke();

    if (_grabbedStroke == null) {
      final size = MediaQuery.of(context).size;
      final tipPx = Offset(tip.dx * size.width, tip.dy * size.height);

      for (final s in _drawController.strokes.reversed) {
        if (s.contains(tipPx, size, 40)) {
          _grabbedStroke = s;
          _lastGrabPos = tip;
          break;
        }
      }
    } else {
      final dx = tip.dx - _lastGrabPos!.dx;
      final dy = tip.dy - _lastGrabPos!.dy;

      for (var i = 0; i < _grabbedStroke!.points.length; i++) {
        final p = _grabbedStroke!.points[i];
        _grabbedStroke!.points[i] = DrawingPoint(x: p.x + dx, y: p.y + dy);
      }
      _lastGrabPos = tip;
      _drawController.notify();
    }
  }

  void _finishStroke() {
    _activeStroke = null;
  }

  Future<void> _share() async {
    final image = await _screenshotController.capture();
    if (image == null) return;

    final directory = await getTemporaryDirectory();
    final imagePath = await File(
            '${directory.path}/neon_art_${DateTime.now().millisecondsSinceEpoch}.png')
        .create();
    await imagePath.writeAsBytes(image);

    // ignore: deprecated_member_use
    await Share.shareXFiles([XFile(imagePath.path)],
        text: 'Check out my Neon Studio creation!');
  }

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;

    return Scaffold(
      backgroundColor: const Color(0xFF0B0D10),
      body: Stack(
        children: [
          // 1. Camera (Native Platform View)
          Positioned.fill(
            child: HandtrackView(
              config: _cfg,
              onFrame: _onFrame,
              onError: (e) => setState(() => _status = "Error: $e"),
            ),
          ),

          // 2. Drawing Content for Screenshot
          Positioned.fill(
            child: Screenshot(
              controller: _screenshotController,
              child: Stack(
                children: [
                  Positioned.fill(
                    child: CustomPaint(
                      painter: _CanvasPainter(_drawController.strokes),
                    ),
                  ),
                ],
              ),
            ),
          ),

          // 3. Landmarks Overlay (Always Visible)
          Positioned.fill(
            child: IgnorePointer(
              child: NeonLandmarksOverlay(frame: _lastFrame),
            ),
          ),

          // 4. UI Overlays
          _buildTopBar(),
          _buildBottomControls(),

          // Floating Gesture Info
          if (_lastFrame != null && _lastFrame!.hands.isNotEmpty)
            Positioned(
              left: _lastFrame!.hands.first.landmarks[8].x * size.width,
              top: (_lastFrame!.hands.first.landmarks[8].y * size.height) - 40,
              child: IgnorePointer(
                child: Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.6),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(
                        color: const Color(0xFF69F0AE).withValues(alpha: 0.3)),
                  ),
                  child: Text(
                    _gestureName,
                    style: const TextStyle(
                        color: Color(0xFF69F0AE),
                        fontSize: 13,
                        fontWeight: FontWeight.bold),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildTopBar() {
    return Positioned(
      top: 50,
      left: 20,
      right: 20,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(20),
        child: BackdropFilter(
          filter: ui.ImageFilter.blur(sigmaX: 12, sigmaY: 12),
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.06),
              border: Border.all(color: Colors.white.withValues(alpha: 0.12)),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Text("Neon Studio",
                          style: TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.w900,
                              fontSize: 18,
                              letterSpacing: 0.5)),
                      Text(
                          "$_status • FPS: ${_lastFrame?.fps.toStringAsFixed(1) ?? "0"}",
                          style: TextStyle(
                              color: Colors.white.withValues(alpha: 0.5),
                              fontSize: 11,
                              fontWeight: FontWeight.w500)),
                    ],
                  ),
                ),
                Row(
                  children: [
                    IconButton(
                      icon: Icon(
                          _voiceController.isListening
                              ? Icons.mic
                              : Icons.mic_none,
                          color: _voiceController.isListening
                              ? Colors.redAccent
                              : Colors.white70),
                      onPressed: () => setState(() {
                        if (_voiceController.isListening) {
                          _voiceController.stopListening();
                        } else {
                          _voiceController.startListening();
                        }
                      }),
                      visualDensity: VisualDensity.compact,
                    ),
                    IconButton(
                        icon: const Icon(Icons.refresh,
                            color: Colors.white70, size: 20),
                        onPressed: _resetEngine,
                        visualDensity: VisualDensity.compact),
                    IconButton(
                        icon: const Icon(Icons.undo,
                            color: Colors.white70, size: 20),
                        onPressed: _drawController.undo,
                        visualDensity: VisualDensity.compact),
                    IconButton(
                        icon: const Icon(Icons.delete_outline,
                            color: Colors.redAccent, size: 20),
                        onPressed: _drawController.clear,
                        visualDensity: VisualDensity.compact),
                  ],
                )
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildBottomControls() {
    return Positioned(
      bottom: 50,
      left: 20,
      right: 20,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          _socialButton(Icons.share_outlined, Colors.cyanAccent, _share),
          const SizedBox(width: 25),
          GestureDetector(
            onTap: () => setState(() {
              _cfg = HandtrackConfig(
                preferFrontCamera: !_cfg.preferFrontCamera,
                targetFps: _cfg.targetFps,
              );
              _status = "Switching Camera...";
            }),
            child: Container(
              width: 64,
              height: 64,
              decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: LinearGradient(
                    colors: [
                      Colors.white.withValues(alpha: 0.15),
                      Colors.white.withValues(alpha: 0.05)
                    ],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  border:
                      Border.all(color: Colors.white.withValues(alpha: 0.2)),
                  boxShadow: const [
                    BoxShadow(
                        color: Colors.black26, blurRadius: 10, spreadRadius: 1)
                  ]),
              child: const Icon(Icons.flip_camera_ios,
                  color: Colors.white, size: 30),
            ),
          ),
        ],
      ),
    );
  }

  Widget _socialButton(IconData icon, Color color, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.1),
          border: Border.all(color: color.withValues(alpha: 0.25)),
          borderRadius: BorderRadius.circular(18),
        ),
        child: Icon(icon, color: color, size: 26),
      ),
    );
  }
}

class _CanvasPainter extends CustomPainter {
  final List<DrawingStroke> strokes;
  _CanvasPainter(this.strokes);

  @override
  void paint(Canvas canvas, Size size) {
    for (final s in strokes) {
      if (s.points.length < 2) continue;

      final paint = Paint()
        ..color = s.color
        ..strokeWidth = s.thickness
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round
        ..style = PaintingStyle.stroke;

      final path = Path();
      path.moveTo(s.points[0].x * size.width, s.points[0].y * size.height);
      for (var i = 1; i < s.points.length; i++) {
        path.lineTo(s.points[i].x * size.width, s.points[i].y * size.height);
      }

      if (s.glow) {
        // Draw the neon glow effect
        canvas.drawPath(
            path,
            paint
              ..maskFilter = const ui.MaskFilter.blur(ui.BlurStyle.normal, 10)
              ..color = s.color.withValues(alpha: 0.6));
        canvas.drawPath(
            path,
            paint
              ..maskFilter = const ui.MaskFilter.blur(ui.BlurStyle.normal, 5)
              ..color = s.color.withValues(alpha: 0.8));
      }

      // Draw core line (always white-ish for the neon center)
      canvas.drawPath(
          path,
          paint
            ..maskFilter = null
            ..strokeWidth = s.thickness * 0.4
            ..color = Colors.white);
    }
  }

  @override
  bool shouldRepaint(covariant _CanvasPainter oldDelegate) => true;
}
