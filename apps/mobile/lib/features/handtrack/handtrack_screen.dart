import "dart:math" as math;

import "package:flutter/material.dart";
import "package:handtrack_engine/handtrack_engine.dart";

import "widgets/neon_landmarks_overlay.dart";
import "widgets/quality_autopilot.dart";

class HandtrackScreen extends StatefulWidget {
  const HandtrackScreen({super.key});

  @override
  State<HandtrackScreen> createState() => _HandtrackScreenState();
}

class _HandtrackScreenState extends State<HandtrackScreen> {
  HandFrame? _last;
  HandtrackConfig _cfg = const HandtrackConfig(targetFps: 30, preferFrontCamera: true);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0B0D10),
      body: SafeArea(
        child: Stack(
          children: [
            Positioned.fill(
              child: HandtrackView(
                config: _cfg,
                onFrame: (f) => setState(() => _last = f),
                onError: (msg) => debugPrint(msg),
              ),
            ),

            Positioned.fill(
              child: IgnorePointer(
                child: NeonLandmarksOverlay(frame: _last),
              ),
            ),

            Positioned(
              left: 12,
              top: 12,
              child: _Hud(frame: _last),
            ),

            Positioned(
              right: 12,
              top: 12,
              child: _Controls(
                cfg: _cfg,
                onCfg: (c) => setState(() => _cfg = c),
              ),
            ),

            // Auto quality manager: adjust targetFps based on actual fps
            Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              child: QualityAutopilot(
                frame: _last,
                current: _cfg,
                onApply: (next) => setState(() => _cfg = next),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Hud extends StatelessWidget {
  const _Hud({required this.frame});
  final HandFrame? frame;

  @override
  Widget build(BuildContext context) {
    final fps = frame?.fps ?? 0;
    final hands = frame?.hands.length ?? 0;
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.35),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withOpacity(0.12)),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        child: DefaultTextStyle(
          style: const TextStyle(color: Colors.white, fontSize: 12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text("FPS: ${fps.toStringAsFixed(1)}"),
              Text("Hands: $hands"),
            ],
          ),
        ),
      ),
    );
  }
}

class _Controls extends StatelessWidget {
  const _Controls({required this.cfg, required this.onCfg});
  final HandtrackConfig cfg;
  final ValueChanged<HandtrackConfig> onCfg;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        _pillButton(
          label: cfg.preferFrontCamera ? "Front" : "Back",
          onTap: () => onCfg(HandtrackConfig(
            maxHands: cfg.maxHands,
            minDetection: cfg.minDetection,
            minPresence: cfg.minPresence,
            minTracking: cfg.minTracking,
            targetFps: cfg.targetFps,
            preferFrontCamera: !cfg.preferFrontCamera,
            enablePreview: cfg.enablePreview,
            androidDelegate: cfg.androidDelegate,
          )),
        ),
        const SizedBox(height: 8),
        _pillButton(
          label: "Target ${cfg.targetFps}fps",
          onTap: () {
            final next = (cfg.targetFps == 60) ? 30 : (cfg.targetFps == 30) ? 24 : 60;
            onCfg(HandtrackConfig(
              maxHands: cfg.maxHands,
              minDetection: cfg.minDetection,
              minPresence: cfg.minPresence,
              minTracking: cfg.minTracking,
              targetFps: next,
              preferFrontCamera: cfg.preferFrontCamera,
              enablePreview: cfg.enablePreview,
              androidDelegate: cfg.androidDelegate,
            ));
          },
        ),
      ],
    );
  }

  Widget _pillButton({required String label, required VoidCallback onTap}) {
    return GestureDetector(
      onTap: onTap,
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.35),
          borderRadius: BorderRadius.circular(999),
          border: Border.all(color: Colors.white.withOpacity(0.12)),
        ),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          child: Text(label, style: const TextStyle(color: Colors.white, fontSize: 12)),
        ),
      ),
    );
  }
}
