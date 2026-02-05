import 'dart:math' as math;
import 'package:handtrack_engine/handtrack_engine.dart';

enum HandGesture {
  none,
  pointingUp,
  closedFist,
  openPalm,
  thumbUp,
  victory,
}

class GestureProcessor {
  static HandGesture process(Hand hand) {
    if (hand.landmarks.length < 21) return HandGesture.none;

    final landmarks = hand.landmarks;

    // Helper to check if a finger is "extended"
    // We check if the tip is further from the wrist than the PIP joint
    bool isExtended(int tip, int pip, int mcp) {
      final wrist = landmarks[0];
      final dTip = math.sqrt(math.pow(landmarks[tip].x - wrist.x, 2) +
          math.pow(landmarks[tip].y - wrist.y, 2));
      final dPip = math.sqrt(math.pow(landmarks[pip].x - wrist.x, 2) +
          math.pow(landmarks[pip].y - wrist.y, 2));
      return dTip > dPip * 1.1; // 10% buffer
    }

    final thumbExtended = isExtended(4, 3, 2);
    final indexExtended = isExtended(8, 6, 5);
    final middleExtended = isExtended(12, 10, 9);
    final ringExtended = isExtended(16, 14, 13);
    final pinkyExtended = isExtended(20, 18, 17);

    // 1. Pointing Up: Only Index extended
    if (indexExtended && !middleExtended && !ringExtended && !pinkyExtended) {
      return HandGesture.pointingUp;
    }

    // 2. Victory: Index and Middle extended
    if (indexExtended && middleExtended && !ringExtended && !pinkyExtended) {
      return HandGesture.victory;
    }

    // 3. Open Palm: All fingers extended
    if (indexExtended && middleExtended && ringExtended && pinkyExtended) {
      return HandGesture.openPalm;
    }

    // 4. Closed Fist: No fingers extended (except maybe thumb)
    if (!indexExtended && !middleExtended && !ringExtended && !pinkyExtended) {
      return HandGesture.closedFist;
    }

    // 5. Thumb Up: Only thumb extended + horizontal/vertical check
    if (thumbExtended &&
        !indexExtended &&
        !middleExtended &&
        !ringExtended &&
        !pinkyExtended) {
      // Check if thumb tip is significantly higher than thumb mcp
      if (landmarks[4].y < landmarks[2].y - 0.05) {
        return HandGesture.thumbUp;
      }
    }

    return HandGesture.none;
  }

  static String gestureName(HandGesture g) {
    switch (g) {
      case HandGesture.pointingUp:
        return "Pointing_Up";
      case HandGesture.closedFist:
        return "Closed_Fist";
      case HandGesture.openPalm:
        return "Open_Palm";
      case HandGesture.thumbUp:
        return "Thumb_Up";
      case HandGesture.victory:
        return "Victory";
      default:
        return "None";
    }
  }
}
