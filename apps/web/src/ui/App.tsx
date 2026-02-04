import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { FilesetResolver, GestureRecognizer, DrawingUtils } from "@mediapipe/tasks-vision";

/**
 * Neon Studio — single-file, production-grade demo (optimized)
 *
 * Goals:
 * - Reliable boot (camera + MediaPipe) with strict-mode safe effects
 * - No illegal invocation (requestVideoFrameCallback bound correctly)
 * - No hook misuse (no hooks inside hooks)
 * - Smooth interaction:
 *    • Point (index extended) → draw
 *    • Pinch (thumb+index) → grab + move selected stroke
 *    • While pinching: rotate by twisting pinch line, scale by hand distance (closer/farther)
 *    • Two pinches → translate + scale + rotate (more stable than single-hand)
 * - Shapes (circle/square/triangle/heart/star/arrow) are strokes too → pickable/movable/scalable
 * - Voice: auto-pick non-Microsoft TTS voice, no voice list UI
 *
 * Notes on speed:
 * - The main cost is client-side vision inference (not your backend). MediaPipe runs in-browser (WASM/WebGL).
 * - You *can* replace the model, but a custom model usually won’t beat MediaPipe out-of-the-box unless you invest in
 *   training + mobile-friendly inference (TFLite/CoreML/NNAPI) and strong optimization.
 */

type Pt = { x: number; y: number };
type StrokePoint = { x: number; y: number; t: number; w: number };
type BBox = { minX: number; minY: number; maxX: number; maxY: number };
type Stroke = { id: string; color: string; points: StrokePoint[]; bbox: BBox };
type InferencePreset = "FAST" | "BALANCED" | "HIGH";

const IDX = {
  wrist: 0,
  thumbTip: 4,
  indexMcp: 5,
  indexPip: 6,
  indexTip: 8,
  middleMcp: 9,
  middlePip: 10,
  middleTip: 12,
  ringMcp: 13,
  ringPip: 14,
  ringTip: 16,
  pinkyMcp: 17,
  pinkyPip: 18,
  pinkyTip: 20,
} as const;

/* ------------------------------ utils ------------------------------ */
function clamp(v: number, a: number, b: number) {
  return Math.max(a, Math.min(b, v));
}
function dist(a: Pt, b: Pt) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
function uid() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16);
}
function emptyBBox(): BBox {
  return { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
}
function expandBBox(bb: BBox, p: Pt) {
  bb.minX = Math.min(bb.minX, p.x);
  bb.minY = Math.min(bb.minY, p.y);
  bb.maxX = Math.max(bb.maxX, p.x);
  bb.maxY = Math.max(bb.maxY, p.y);
}
function padBBox(bb: BBox, pad: number): BBox {
  return { minX: bb.minX - pad, minY: bb.minY - pad, maxX: bb.maxX + pad, maxY: bb.maxY + pad };
}
function bboxContains(bb: BBox, p: Pt) {
  return p.x >= bb.minX && p.x <= bb.maxX && p.y >= bb.minY && p.y <= bb.maxY;
}
function centroid(pts: Pt[]): Pt {
  if (!pts.length) return { x: 0, y: 0 };
  let sx = 0,
    sy = 0;
  for (const p of pts) {
    sx += p.x;
    sy += p.y;
  }
  return { x: sx / pts.length, y: sy / pts.length };
}
function rotatePoint(p: Pt, c: Pt, ang: number): Pt {
  const s = Math.sin(ang);
  const co = Math.cos(ang);
  const x = p.x - c.x;
  const y = p.y - c.y;
  return { x: c.x + x * co - y * s, y: c.y + x * s + y * co };
}
function scalePoint(p: Pt, c: Pt, k: number): Pt {
  return { x: c.x + (p.x - c.x) * k, y: c.y + (p.y - c.y) * k };
}
function normAngleDelta(d: number) {
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return d;
}
function distToSegment(p: Pt, a: Pt, b: Pt) {
  const vx = b.x - a.x,
    vy = b.y - a.y;
  const wx = p.x - a.x,
    wy = p.y - a.y;
  const c1 = vx * wx + vy * wy;
  if (c1 <= 0) return Math.hypot(p.x - a.x, p.y - a.y);
  const c2 = vx * vx + vy * vy;
  if (c2 <= c1) return Math.hypot(p.x - b.x, p.y - b.y);
  const t = c1 / c2;
  const px = a.x + t * vx,
    py = a.y + t * vy;
  return Math.hypot(p.x - px, p.y - py);
}

/* --------------------------- One-Euro smoothing -------------------------- */
class LowPass {
  private y = 0;
  private a = 0;
  private s = 0;
  constructor(alpha: number) {
    this.setAlpha(alpha);
  }
  setAlpha(alpha: number) {
    this.a = clamp(alpha, 0, 1);
  }
  filter(x: number) {
    const y = this.a * x + (1 - this.a) * this.s;
    this.s = y;
    this.y = y;
    return y;
  }
  lastValue() {
    return this.y;
  }
}
function alpha(cutoff: number, dt: number) {
  const tau = 1.0 / (2 * Math.PI * cutoff);
  return 1.0 / (1.0 + tau / dt);
}
class OneEuro2D {
  private x = new LowPass(1);
  private y = new LowPass(1);
  private dx = new LowPass(1);
  private dy = new LowPass(1);
  private lastT = 0;
  constructor(private minCutoff = 0.9, private beta = 0.02, private dCutoff = 1.0) {}
  set(minCutoff: number, beta: number) {
    this.minCutoff = minCutoff;
    this.beta = beta;
  }
  filter(p: Pt, tMs: number): Pt {
    if (this.lastT === 0) {
      this.lastT = tMs;
      this.x.filter(p.x);
      this.y.filter(p.y);
      return p;
    }
    const dt = Math.max(1 / 120, (tMs - this.lastT) / 1000);
    this.lastT = tMs;

    const prevX = this.x.lastValue();
    const prevY = this.y.lastValue();
    const vx = (p.x - prevX) / dt;
    const vy = (p.y - prevY) / dt;

    this.dx.setAlpha(alpha(this.dCutoff, dt));
    this.dy.setAlpha(alpha(this.dCutoff, dt));
    const edx = this.dx.filter(vx);
    const edy = this.dy.filter(vy);

    const cutoffX = this.minCutoff + this.beta * Math.abs(edx);
    const cutoffY = this.minCutoff + this.beta * Math.abs(edy);

    this.x.setAlpha(alpha(cutoffX, dt));
    this.y.setAlpha(alpha(cutoffY, dt));

    return { x: this.x.filter(p.x), y: this.y.filter(p.y) };
  }
}

/* ------------------------------ detection types ------------------------------ */
type HandDet = {
  frameIndex: number;
  handedness: "Left" | "Right" | "Unknown";
  wrist: Pt;
  palm: Pt;
  indexTip: Pt;
  indexPip: Pt;
  thumbTip: Pt;
  indexMcp: Pt;
  middleMcp: Pt;
  middleTip: Pt;
  ringTip: Pt;
  pinkyTip: Pt;
  middlePip: Pt;
  ringPip: Pt;
  pinkyPip: Pt;

  scalePx: number; // approx distance proxy (hand size in pixels)
  pinchStrength: number; // 0..1

  rawPoint: boolean;
  rawPalm: boolean;
  rawGrab: boolean;
  rawLandmarks: Pt[];
  gesture?: string;
  gestureConfidence?: number;
};

type Track = {
  id: string;
  lastSeenMs: number;
  det: HandDet;

  pointFilter: OneEuro2D;
  pinchFilter: OneEuro2D;
  palmFilter: OneEuro2D;

  stablePoint: number;
  stablePalm: number;

  isPointing: boolean;
  isPinching: boolean;
  isOpenPalm: boolean;

  drawStrokeId: string;
  selectedStrokeId: string;

  pinchStartScalePx: number;
  pinchStartAng: number;
  lastPinchAng: number;
  lastPinchPt: Pt;
};

function getInferSize(preset: InferencePreset) {
  if (preset === "FAST") return { w: 640, h: 360 };
  if (preset === "BALANCED") return { w: 960, h: 540 };
  return { w: 1280, h: 720 };
}

/* ------------------------------ gestures ------------------------------ */
function isFingerExtended(wrist: Pt, tip: Pt, pip: Pt, margin: number) {
  return dist(wrist, tip) > dist(wrist, pip) + margin;
}
function computeGestureBooleans(det: Omit<HandDet, "rawPoint" | "rawPalm" | "rawGrab" | "rawLandmarks">): Pick<HandDet, "rawPoint" | "rawPalm"> {
  const margin = clamp(det.scalePx * 0.06, 6, 18);
  const indexExt = isFingerExtended(det.wrist, det.indexTip, det.indexPip, margin);
  const middleExt = isFingerExtended(det.wrist, det.middleTip, det.middlePip, margin);
  const ringExt = isFingerExtended(det.wrist, det.ringTip, det.ringPip, margin);
  const pinkyExt = isFingerExtended(det.wrist, det.pinkyTip, det.pinkyPip, margin);

  const rawPoint = indexExt && !middleExt && !ringExt && !pinkyExt && det.pinchStrength < 0.25;
  const rawPalm = indexExt && middleExt && ringExt && pinkyExt && det.pinchStrength < 0.2;

  return { rawPoint, rawPalm };
}

/* ------------------------------ voice helpers ------------------------------ */
function parseCssColorCandidate(raw: string) {
  const s = raw.trim().toLowerCase();
  const named: Record<string, string> = {
    black: "#000000",
    white: "#ffffff",
    red: "#ff3333",
    green: "#33ff66",
    blue: "#3388ff",
    yellow: "#ffd84d",
    purple: "#b366ff",
    pink: "#ff66cc",
    orange: "#ff9933",
    cyan: "#55ddff",
  };
  if (s in named) return named[s];
  if (/^#([0-9a-f]{3}|[0-9a-f]{6})$/i.test(s)) return s;
  return s;
}
function parseSpokenNumber(text: string): number | null {
  const m = text.match(/(-?\d+(\.\d+)?)/);
  if (m?.[1]) return parseFloat(m[1]);

  const w = text.toLowerCase().replace(/-/g, " ").split(/\s+/).filter(Boolean);
  if (!w.length) return null;

  const ones: Record<string, number> = { zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9 };
  const teens: Record<string, number> = { ten: 10, eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19 };
  const tens: Record<string, number> = { twenty: 20, thirty: 30, forty: 40, fifty: 50, sixty: 60, seventy: 70, eighty: 80, ninety: 90 };

  function parseIntPhrase(tokens: string[]) {
    let i = 0;
    let val = 0;
    while (i < tokens.length) {
      const t = tokens[i];
      if (t in ones) { val += ones[t]; i++; }
      else if (t in teens) { val += teens[t]; i++; }
      else if (t in tens) {
        val += tens[t];
        if (i + 1 < tokens.length && tokens[i + 1] in ones) { val += ones[tokens[i + 1]]; i += 2; }
        else i++;
      } else if (t === "and") i++;
      else break;
    }
    return { val, consumed: i };
  }

  const pointIdx = w.indexOf("point");
  if (pointIdx === -1) {
    const { val, consumed } = parseIntPhrase(w);
    return consumed ? val : null;
  }
  const A = parseIntPhrase(w.slice(0, pointIdx)).val;
  const decimals = clamp(parseIntPhrase(w.slice(pointIdx + 1)).val, 0, 99);
  return parseFloat(`${A}.${decimals.toString().padStart(2, "0")}`);
}

/* ------------------------------- main component ------------------------------ */
export default function App() {
  const theme = useMemo(
    () => ({
      bg: "#0B0D10",
      panel: "rgba(0,0,0,0.38)",
      border: "rgba(255,255,255,0.12)",
      border2: "rgba(255,255,255,0.18)",
      fg: "rgba(255,255,255,0.92)",
      muted: "rgba(255,255,255,0.68)",
      accent: "#69F0AE",
      warn: "#ffcc66",
      danger: "#ff5566",
    }),
    []
  );

  const fontStack =
    'Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji"';

  /* ------------------------------- DOM refs ------------------------------- */
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const drawRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const inferCanvasRef = useRef<HTMLCanvasElement | null>(null);

  /* -------------------------- MediaPipe landmarker ------------------------- */
  const [landmarker, setLandmarker] = useState<GestureRecognizer | null>(null);

  /* --------------------------- core runtime refs --------------------------- */
  const strokesRef = useRef<Stroke[]>([]);
  const redoRef = useRef<Stroke[]>([]);
  const tracksRef = useRef<Track[]>([]);
  const needsFullRedrawRef = useRef(false);

  // admin/guest control
  const adminScaleRef = useRef<number>(0);
  const guestUnlockUntilRef = useRef<number>(0);
  const adminTolerance = 0.22;

  // two-hand transform state
  const twoPinchRef = useRef({
    active: false,
    strokeId: "",
    baseDist: 1,
    baseAng: 0,
    lastK: 1,
    lastAng: 0,
    lastMid: { x: 0, y: 0 } as Pt,
  });

  /* ------------------------------- settings ------------------------------- */
  const settingsRef = useRef({
    showLandmarks: true,
    glow: true,
    brushColor: "#69F0AE",
    baseThickness: 6,

    inferPreset: "FAST" as InferencePreset,
    inferFpsCap: 24,
    maxHands: 4,

    oneEuroMinCutoff: 0.9,
    oneEuroBeta: 0.02,

    pickRadiusPx: 42,
  });

  /* ---------------------------------- HUD -------------------------------- */
  const [ready, setReady] = useState(false);
  const [loadingStep, setLoadingStep] = useState("Initializing…");
  const [hud, setHud] = useState({ fps: 0, inferMs: 0, strokes: 0, hands: 0, guestUnlocked: false });

  /* -------------------------------- voice -------------------------------- */
  const [voiceOn, setVoiceOn] = useState(true);
  const voiceOnRef = useRef(true);
  useEffect(() => {
    voiceOnRef.current = voiceOn;
  }, [voiceOn]);

  const [voiceHint, setVoiceHint] = useState(
    'Mic on. Try: "change color to red", "glow off", "thickness 10", "draw small heart", "export".'
  );

  const speechRecRef = useRef<any>(null);
  const voiceRef = useRef<SpeechSynthesisVoice | null>(null);
  const ttsGuardRef = useRef({
    speaking: false,
    lastSpoken: "",
    ignoreUntil: 0,
    suppressAutoRestart: false,
    lastCmd: { text: "", at: 0 },
  });

  const handleVoiceCommandRef = useRef<((t: string) => void) | null>(null);

  // Inject a nicer font (Inter) if the project doesn't already provide it.
  useEffect(() => {
    const id = "neon-font-inter";
    if (document.getElementById(id)) return;
    const link = document.createElement("link");
    link.id = id;
    link.rel = "stylesheet";
    link.href = "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap";
    document.head.appendChild(link);
    return () => link.remove();
  }, []);

  useEffect(() => {
    const pick = () => {
      const all = window.speechSynthesis.getVoices();
      if (!all?.length) return;

      const nonMs = all.filter((v) => !/^microsoft/i.test(v.name));
      const best =
        nonMs.find((v) => /google|natural|neural/i.test(v.name.toLowerCase())) ||
        nonMs.find((v) => v.lang?.toLowerCase().startsWith("en")) ||
        nonMs[0] ||
        all.find((v) => v.lang?.toLowerCase().startsWith("en")) ||
        all[0] ||
        null;

      voiceRef.current = best;
    };

    pick();
    window.speechSynthesis.onvoiceschanged = pick;
    return () => {
      window.speechSynthesis.onvoiceschanged = null;
    };
  }, []);

  const speak = useCallback((msg: string) => {
    if (!voiceOnRef.current) return;

    const rec = speechRecRef.current;
    const guard = ttsGuardRef.current;

    const normalized = msg.toLowerCase().trim();
    guard.lastSpoken = normalized;
    guard.speaking = true;
    guard.ignoreUntil = performance.now() + 10000; // Safe high value, onend will reset it
    window.speechSynthesis.cancel();

    const u = new SpeechSynthesisUtterance(msg);
    u.rate = 1.0;
    u.pitch = 1.0;
    if (voiceRef.current) u.voice = voiceRef.current;

    u.onend = () => {
      guard.speaking = false;
      guard.suppressAutoRestart = false;
      // Wait a bit before allowing recognition to resume listening to surroundings
      setTimeout(() => {
        guard.ignoreUntil = performance.now() + 100;
      }, 500);
    };

    window.speechSynthesis.speak(u);
  }, []);

  /* ------------------------------ canvas sizing ------------------------------ */
  const resizeCanvasesToVideo = useCallback(() => {
    const v = videoRef.current;
    const draw = drawRef.current;
    const ov = overlayRef.current;
    if (!v || !draw || !ov) return;

    const w = v.videoWidth || 1280;
    const h = v.videoHeight || 720;

    if (draw.width !== w || draw.height !== h) {
      draw.width = w;
      draw.height = h;
      needsFullRedrawRef.current = true;
    }
    if (ov.width !== w || ov.height !== h) {
      ov.width = w;
      ov.height = h;
    }
  }, []);

  /* ------------------------------ stroke rendering ------------------------------ */
  const paintStrokeSegment = useCallback(
    (ctx: CanvasRenderingContext2D, color: string, a: StrokePoint, b: StrokePoint, glow: boolean) => {
      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = color;
      ctx.lineWidth = b.w;

      if (glow) {
        ctx.shadowBlur = 18;
        ctx.shadowColor = color;
      } else {
        ctx.shadowBlur = 0;
      }

      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      ctx.restore();
    },
    []
  );

  const redrawAll = useCallback(() => {
    const canvas = drawRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const { glow } = settingsRef.current;

    for (const s of strokesRef.current) {
      if (s.points.length < 2) continue;
      for (let i = 1; i < s.points.length; i++) {
        paintStrokeSegment(ctx, s.color, s.points[i - 1], s.points[i], glow);
      }
    }
  }, [paintStrokeSegment]);

  const startStroke = useCallback((color: string, p: Pt, t: number, thickness: number): string => {
    redoRef.current = [];
    const bb = emptyBBox();
    expandBBox(bb, p);
    const s: Stroke = { id: uid(), color, points: [{ x: p.x, y: p.y, t, w: thickness }], bbox: bb };
    strokesRef.current = [...strokesRef.current, s];
    return s.id;
  }, []);

  const addStrokePoint = useCallback(
    (strokeId: string, p: Pt, t: number, thickness: number) => {
      const strokes = strokesRef.current;
      const idx = strokes.findIndex((s) => s.id === strokeId);
      if (idx === -1) return;

      const s = strokes[idx];
      const sp: StrokePoint = { x: p.x, y: p.y, t, w: thickness };
      s.points.push(sp);
      expandBBox(s.bbox, p);

      const canvas = drawRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d")!;
      const { glow } = settingsRef.current;

      if (s.points.length >= 2) {
        const a = s.points[s.points.length - 2];
        const b = s.points[s.points.length - 1];
        paintStrokeSegment(ctx, s.color, a, b, glow);
      }
    },
    [paintStrokeSegment]
  );

  const clearAll = useCallback(() => {
    strokesRef.current = [];
    redoRef.current = [];
    needsFullRedrawRef.current = true;
    speak("Cleared.");
  }, [speak]);

  const undo = useCallback(() => {
    const st = strokesRef.current;
    if (!st.length) return;
    const last = st[st.length - 1];
    redoRef.current = [last, ...redoRef.current];
    strokesRef.current = st.slice(0, -1);
    needsFullRedrawRef.current = true;
    speak("Undo.");
  }, [speak]);

  const redo = useCallback(() => {
    const r = redoRef.current;
    if (!r.length) return;
    const [first, ...rest] = r;
    redoRef.current = rest;
    strokesRef.current = [...strokesRef.current, first];
    needsFullRedrawRef.current = true;
    speak("Redo.");
  }, [speak]);

  const pickStroke = useCallback((p: Pt): string | "" => {
    const pad = settingsRef.current.pickRadiusPx;
    let bestId = "";
    let bestD = Infinity;

    for (const s of strokesRef.current) {
      const bb = padBBox(s.bbox, pad);
      if (!bboxContains(bb, p)) continue;

      for (let i = 1; i < s.points.length; i++) {
        const a = s.points[i - 1];
        const b = s.points[i];
        const d = distToSegment(p, a, b);
        if (d < bestD) {
          bestD = d;
          bestId = s.id;
        }
      }
    }

    return bestD < pad ? bestId : "";
  }, []);

  const translateStroke = useCallback((strokeId: string, dx: number, dy: number) => {
    strokesRef.current = strokesRef.current.map((s) => {
      if (s.id !== strokeId) return s;
      const bb = emptyBBox();
      const pts = s.points.map((pt) => {
        const np = { ...pt, x: pt.x + dx, y: pt.y + dy };
        expandBBox(bb, np);
        return np;
      });
      return { ...s, points: pts, bbox: bb };
    });
    needsFullRedrawRef.current = true;
  }, []);

  const rotateStroke = useCallback((strokeId: string, dAng: number) => {
    strokesRef.current = strokesRef.current.map((s) => {
      if (s.id !== strokeId) return s;
      const c = centroid(s.points.map((p) => ({ x: p.x, y: p.y })));
      const bb = emptyBBox();
      const pts = s.points.map((pt) => {
        const rp = rotatePoint({ x: pt.x, y: pt.y }, c, dAng);
        const np = { ...pt, x: rp.x, y: rp.y };
        expandBBox(bb, np);
        return np;
      });
      return { ...s, points: pts, bbox: bb };
    });
    needsFullRedrawRef.current = true;
  }, []);

  const scaleStroke = useCallback((strokeId: string, k: number) => {
    const kk = clamp(k, 0.25, 3.0);
    strokesRef.current = strokesRef.current.map((s) => {
      if (s.id !== strokeId) return s;
      const c = centroid(s.points.map((p) => ({ x: p.x, y: p.y })));
      const bb = emptyBBox();
      const pts = s.points.map((pt) => {
        const sp = scalePoint({ x: pt.x, y: pt.y }, c, kk);
        const np = { ...pt, x: sp.x, y: sp.y, w: clamp(pt.w * kk, 1, 40) };
        expandBBox(bb, np);
        return np;
      });
      return { ...s, points: pts, bbox: bb };
    });
    needsFullRedrawRef.current = true;
  }, []);

  /* ------------------------------ shapes ------------------------------ */
  const drawShapeAtCenter = useCallback(
    (shape: "circle" | "square" | "triangle" | "heart" | "star" | "arrow", size: "small" | "medium" | "big", color: string) => {
      const canvas = drawRef.current;
      if (!canvas) return;

      const minDim = Math.min(canvas.width, canvas.height);
      const r = size === "small" ? minDim * 0.11 : size === "medium" ? minDim * 0.17 : minDim * 0.24;
      const c: Pt = { x: canvas.width / 2, y: canvas.height / 2 };
      const t0 = performance.now();
      const thick = settingsRef.current.baseThickness;

      const id = startStroke(color, c, t0, thick);

      const pts: Pt[] = [];
      if (shape === "circle") {
        const N = 72;
        for (let i = 0; i <= N; i++) {
          const a = (i / N) * Math.PI * 2;
          pts.push({ x: c.x + Math.cos(a) * r, y: c.y + Math.sin(a) * r });
        }
      } else if (shape === "square") {
        pts.push({ x: c.x - r, y: c.y - r });
        pts.push({ x: c.x + r, y: c.y - r });
        pts.push({ x: c.x + r, y: c.y + r });
        pts.push({ x: c.x - r, y: c.y + r });
        pts.push({ x: c.x - r, y: c.y - r });
      } else if (shape === "triangle") {
        pts.push({ x: c.x, y: c.y - r });
        pts.push({ x: c.x + r, y: c.y + r });
        pts.push({ x: c.x - r, y: c.y + r });
        pts.push({ x: c.x, y: c.y - r });
      } else if (shape === "heart") {
        const N = 96;
        for (let i = 0; i <= N; i++) {
          const tt = (i / N) * Math.PI * 2;
          const x = 16 * Math.pow(Math.sin(tt), 3);
          const y = 13 * Math.cos(tt) - 5 * Math.cos(2 * tt) - 2 * Math.cos(3 * tt) - Math.cos(4 * tt);
          pts.push({ x: c.x + (x / 18) * r, y: c.y - (y / 18) * r });
        }
      } else if (shape === "star") {
        const spikes = 5;
        const outer = r;
        const inner = r * 0.45;
        for (let i = 0; i <= spikes * 2; i++) {
          const rr = i % 2 === 0 ? outer : inner;
          const a = (i / (spikes * 2)) * Math.PI * 2 - Math.PI / 2;
          pts.push({ x: c.x + Math.cos(a) * rr, y: c.y + Math.sin(a) * rr });
        }
        pts.push(pts[0]);
      } else if (shape === "arrow") {
        const w = r * 0.95;
        const h = r * 0.55;
        pts.push({ x: c.x - w, y: c.y });
        pts.push({ x: c.x + w * 0.35, y: c.y });
        pts.push({ x: c.x + w * 0.35, y: c.y - h });
        pts.push({ x: c.x + w, y: c.y });
        pts.push({ x: c.x + w * 0.35, y: c.y + h });
        pts.push({ x: c.x + w * 0.35, y: c.y });
        pts.push({ x: c.x - w, y: c.y });
      }

      for (let i = 0; i < pts.length; i++) addStrokePoint(id, pts[i], t0 + i, thick);

      speak(`${size} ${shape} drawn`);
    },
    [addStrokePoint, speak, startStroke]
  );

  /* ------------------------------ export helpers ------------------------------ */
  const exportBlobMirrored = useCallback(async (): Promise<Blob | null> => {
    const canvas = drawRef.current;
    if (!canvas) return null;

    const tmp = document.createElement("canvas");
    tmp.width = canvas.width;
    tmp.height = canvas.height;
    const tctx = tmp.getContext("2d")!;
    tctx.save();
    tctx.translate(tmp.width, 0);
    tctx.scale(-1, 1);
    tctx.drawImage(canvas, 0, 0);
    tctx.restore();

    return await new Promise((resolve) => tmp.toBlob((b) => resolve(b), "image/png"));
  }, []);

  const copyToClipboard = useCallback(async () => {
    const blob = await exportBlobMirrored();
    if (!blob) return;
    try {
      const data = [new ClipboardItem({ [blob.type]: blob })];
      await navigator.clipboard.write(data);
      speak("Image copied to clipboard.");
      setVoiceHint("Image copied! Now you can paste it anywhere.");
    } catch (err) {
      console.error("Clipboard copy failed:", err);
      setVoiceHint("Couldn't copy image. Try Export PNG.");
    }
  }, [exportBlobMirrored, speak]);

  const downloadPng = useCallback(async () => {
    const blob = await exportBlobMirrored();
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `neon_export_${Date.now()}.png`;
    a.click();
    URL.revokeObjectURL(url);
    speak("Exported.");
  }, [exportBlobMirrored, speak]);

  const sharePng = useCallback(async (target?: "whatsapp" | "instagram" | "facebook" | "x") => {
    const blob = await exportBlobMirrored();
    if (!blob) return;

    if (target) {
      await copyToClipboard();
      setTimeout(() => {
        let url = "";
        if (target === "whatsapp") url = "https://wa.me/";
        else if (target === "instagram") url = "https://www.instagram.com/";
        else if (target === "facebook") url = "https://www.facebook.com/";
        else if (target === "x") url = "https://twitter.com/intent/tweet";
        
        if (url) window.open(url, "_blank");
      }, 500);
      return;
    }

    const file = new File([blob], `neon_export_${Date.now()}.png`, { type: "image/png" });
    const nav = navigator as any;

    try {
      if (nav.canShare?.({ files: [file] }) && nav.share) {
        await nav.share({ title: "Neon Studio", text: "Made with Neon Studio", files: [file] });
        speak("Sharing...");
      } else {
        await downloadPng();
        setVoiceHint("Native sharing unavailable; downloaded instead.");
      }
    } catch (e: any) {
      if (e.name !== "AbortError") {
        setVoiceHint("Share failed. Try Export PNG.");
        await downloadPng();
      }
    }
  }, [copyToClipboard, downloadPng, exportBlobMirrored, speak]);

  /* ------------------------------ voice commands ------------------------------ */
  const effectiveColorForHand = useCallback((isAdmin: boolean) => {
    if (!isAdmin && Date.now() < guestUnlockUntilRef.current) return "#000000";
    return settingsRef.current.brushColor;
  }, []);

  const handleVoiceCommand = useCallback(
    (text: string) => {
      const t = text.toLowerCase();

      if ((t.includes("unlock") && (t.includes("guest") || t.includes("others"))) || t.includes("everyone can draw")) {
        guestUnlockUntilRef.current = Date.now() + 30_000;
        speak("Guests unlocked for thirty seconds. Guests draw in black.");
        return;
      }
      if ((t.includes("lock") && (t.includes("guest") || t.includes("others"))) || t.includes("lock guests")) {
        guestUnlockUntilRef.current = 0;
        speak("Guests locked.");
        return;
      }

      if (t.includes("glow on")) {
        settingsRef.current.glow = true;
        needsFullRedrawRef.current = true;
        speak("Glow on.");
        return;
      }
      if (t.includes("glow off")) {
        settingsRef.current.glow = false;
        needsFullRedrawRef.current = true;
        speak("Glow off.");
        return;
      }

      if (t.includes("thickness") || t.includes("brush size")) {
        const num = parseSpokenNumber(t);
        if (num != null) {
          const v = clamp(num, 1, 40);
          settingsRef.current.baseThickness = v;
          speak(`Thickness ${v.toFixed(0)}`);
        } else {
          speak("I couldn't understand the thickness value.");
        }
        return;
      }

      const colorMatch = t.match(/(change|set)\s+(the\s+)?color(\s+to)?\s+(.+)$/i);
      if (colorMatch?.[4]) {
        const c = parseCssColorCandidate(colorMatch[4]);
        settingsRef.current.brushColor = c;
        speak("Color changed.");
        return;
      }

      if (t.includes("draw")) {
        const size = t.includes("small") ? "small" : t.includes("big") || t.includes("large") ? "big" : "medium";

        const shape =
          t.includes("circle") ? "circle" :
          t.includes("square") ? "square" :
          t.includes("triangle") ? "triangle" :
          t.includes("heart") ? "heart" :
          t.includes("star") ? "star" :
          t.includes("arrow") ? "arrow" :
          null;

        if (shape) {
          let color = settingsRef.current.brushColor;
          if (t.includes("black")) color = "#000000";
          if (t.includes("white")) color = "#ffffff";
          drawShapeAtCenter(shape, size, color);
          return;
        }
      }

      if (t.includes("export") || t.includes("save") || t.includes("download")) return void downloadPng();
      if (t.includes("share")) return void sharePng();
      if (t.includes("copy")) return void copyToClipboard();
      if (t.includes("undo")) return undo();
      if (t.includes("redo")) return redo();
      if (t.includes("clear")) return clearAll();

      speak("No recognized command.");
      setVoiceHint('Try: "change color to red", "glow off", "thickness 12", "draw small star", "unlock guests", "export".');
    },
    [clearAll, drawShapeAtCenter, downloadPng, redo, speak, undo, copyToClipboard, sharePng]
  );

  const resetVoice = useCallback(() => {
    try {
      speechRecRef.current?.abort();
    } catch {}
    speechRecRef.current = null;
    setVoiceOn(false);
    setTimeout(() => setVoiceOn(true), 100);
    speak("Voice system reset.");
  }, [speak]);

  useEffect(() => {
    handleVoiceCommandRef.current = handleVoiceCommand;
  }, [handleVoiceCommand]);

  const setupSpeechRecognition = useCallback(() => {
    const w = window as any;
    const SR = w.SpeechRecognition || w.webkitSpeechRecognition;
    if (!SR) {
      setVoiceHint("Voice not supported. Try Chrome/Edge.");
      return null;
    }

    const rec = new SR();
    rec.lang = "en-US";
    rec.continuous = true;
    rec.interimResults = true;
    rec.maxAlternatives = 1;

    rec.onresult = (event: any) => {
      const guard = ttsGuardRef.current;
      const now = performance.now();
      if (guard.speaking || now < guard.ignoreUntil) return;

      let finalText = "";
      let interim = "";

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const res = event.results[i];
        const txt = res[0].transcript.trim();
        if (res.isFinal) finalText += (finalText ? " " : "") + txt;
        else interim += (interim ? " " : "") + txt;
      }

      const ft = finalText.toLowerCase().trim();
      const echo = guard.lastSpoken && (ft === guard.lastSpoken || ft.includes(guard.lastSpoken));
      if (echo) return;

      if (ft && guard.lastCmd.text === ft && now - guard.lastCmd.at < 1800) return;
      if (ft) guard.lastCmd = { text: ft, at: now };

      if (interim) setVoiceHint(`Heard: "${interim}"`);
      if (finalText) {
        setVoiceHint(`Command: "${finalText}"`);
        handleVoiceCommandRef.current?.(finalText);
      }
    };

    rec.onstart = () => {
      setVoiceHint('Mic active. Try "brush red" or "glow off".');
      console.log("[Voice] Started.");
    };

    rec.onend = () => {
      console.log("[Voice] Ended.");
      if (voiceOnRef.current) {
        try { rec.start(); } catch {}
      }
    };

    rec.onerror = (e: any) => {
      console.warn("[Voice] Error:", e.error);
      if (e.error === "not-allowed") setVoiceHint("Mic blocked. Click Mic once to allow.");
      else if (e.error !== "no-speech") setVoiceHint(`Voice error: ${e.error}`);
    };
    return rec;
  }, []);

  /* ------------------------------ camera + model boot ------------------------------ */
  const startCamera = useCallback(async () => {
    const v = videoRef.current;
    if (!v) return;

    try {
      console.log("[startCamera] Closing old stream if any...");
      const old = v.srcObject as MediaStream | null;
      old?.getTracks().forEach((tr) => tr.stop());
      v.srcObject = null;

      console.log("[startCamera] Requesting getUserMedia...");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      v.muted = true;
      v.playsInline = true;
      v.srcObject = stream;

      console.log("[startCamera] Waiting for loadedmetadata or readyState...");
      // Wait for metadata to resolve dimensions
      if (v.readyState < 1) {
        await new Promise<void>((resolve) => {
          const timeout = setTimeout(() => {
            console.warn("[startCamera] Metadata timeout - proceeding anyway");
            v.removeEventListener("loadedmetadata", onMeta);
            resolve();
          }, 3000);
          const onMeta = () => {
            console.log("[startCamera] Metadata loaded");
            clearTimeout(timeout);
            v.removeEventListener("loadedmetadata", onMeta);
            resolve();
          };
          v.addEventListener("loadedmetadata", onMeta);
        });
      }

      console.log("[startCamera] Playing video...");
      await v.play();
      console.log("[startCamera] Camera ready.");
      resizeCanvasesToVideo();
    } catch (e) {
      console.error("[startCamera] CRITICAL ERROR:", e);
      throw e;
    }
  }, [resizeCanvasesToVideo]);

  /* ------------------------------ track assignment ------------------------------ */
  const assignTracks = useCallback((dets: HandDet[], nowMs: number) => {
    const prev = tracksRef.current.slice();
    const used = new Set<string>();
    const next: Track[] = [];
    const maxMatchDist = 140;

    for (const det of dets) {
      let best: { tr: Track; d: number } | null = null;

      for (const tr of prev) {
        if (used.has(tr.id)) continue;
        const d = dist(tr.det.wrist, det.wrist);
        if (d < maxMatchDist && (!best || d < best.d)) best = { tr, d };
      }

      if (best) {
        used.add(best.tr.id);
        best.tr.lastSeenMs = nowMs;
        best.tr.det = det;
        next.push(best.tr);
      } else {
        const tr: Track = {
          id: uid(),
          lastSeenMs: nowMs,
          det,

          pointFilter: new OneEuro2D(settingsRef.current.oneEuroMinCutoff, settingsRef.current.oneEuroBeta),
          pinchFilter: new OneEuro2D(settingsRef.current.oneEuroMinCutoff, settingsRef.current.oneEuroBeta),
          palmFilter: new OneEuro2D(settingsRef.current.oneEuroMinCutoff, settingsRef.current.oneEuroBeta),

          stablePoint: 0,
          stablePalm: 0,

          isPointing: false,
          isPinching: false,
          isOpenPalm: false,

          drawStrokeId: "",
          selectedStrokeId: "",

          pinchStartScalePx: det.scalePx,
          pinchStartAng: 0,
          lastPinchAng: 0,
          lastPinchPt: det.wrist,
        };
        next.push(tr);
      }
    }

    const ttl = 700;
    const survivors = prev.filter((tr) => nowMs - tr.lastSeenMs < ttl && !used.has(tr.id));
    tracksRef.current = [...next, ...survivors].slice(0, settingsRef.current.maxHands);
  }, []);

  const updateAdminScaleFromTracks = useCallback(() => {
    const tracks = tracksRef.current;
    if (!tracks.length) return;
    const best = tracks.reduce((a, b) => (b.det.scalePx > a.det.scalePx ? b : a));
    const target = best.det.scalePx;
    if (!adminScaleRef.current) adminScaleRef.current = target;
    adminScaleRef.current = adminScaleRef.current * 0.85 + target * 0.15;
  }, []);

  const isAdminGroup = useCallback((tr: Track) => {
    const s = adminScaleRef.current || tr.det.scalePx;
    const rel = Math.abs(tr.det.scalePx - s) / Math.max(1, s);
    return rel <= adminTolerance;
  }, []);

  const guestsUnlocked = useCallback(() => Date.now() < guestUnlockUntilRef.current, []);
  const isAllowedToControl = useCallback((tr: Track) => isAdminGroup(tr) || guestsUnlocked(), [guestsUnlocked, isAdminGroup]);

  const maybeUnlockGuestsByAdminBlackStrokeStart = useCallback(
    (tr: Track, color: string) => {
      if (isAdminGroup(tr) && color.toLowerCase() === "#000000") {
        guestUnlockUntilRef.current = Date.now() + 30_000;
        speak("Guests unlocked for thirty seconds. Guests draw in black.");
      }
    },
    [isAdminGroup, speak]
  );

  /* ------------------------------ interactions ------------------------------ */
  const processInteractions = useCallback(
    (nowMs: number) => {
      const canvas = drawRef.current;
      if (!canvas) return;

      const tracks = tracksRef.current;

      const act = tracks.map((tr) => {
        tr.pointFilter.set(settingsRef.current.oneEuroMinCutoff, settingsRef.current.oneEuroBeta);
        tr.pinchFilter.set(settingsRef.current.oneEuroMinCutoff, settingsRef.current.oneEuroBeta);
        tr.palmFilter.set(settingsRef.current.oneEuroMinCutoff, settingsRef.current.oneEuroBeta);

        const det = tr.det;
        const allowed = isAllowedToControl(tr);

        tr.stablePoint = det.rawPoint ? Math.min(6, tr.stablePoint + 1) : Math.max(0, tr.stablePoint - 1);
        tr.stablePalm = det.rawPalm ? Math.min(6, tr.stablePalm + 1) : Math.max(0, tr.stablePalm - 1);

        const POINT_ACTIVE = tr.stablePoint >= 3;
        const PALM_ACTIVE = tr.stablePalm >= 3;

        const pinchOn = det.pinchStrength > 0.78 || det.rawGrab;
        const pinchOff = det.pinchStrength < 0.55 && !det.rawGrab;

        const pointPt = tr.pointFilter.filter(det.indexTip, nowMs);
        const pinchPt = tr.pinchFilter.filter(
          { x: (det.indexTip.x + det.thumbTip.x) / 2, y: (det.indexTip.y + det.thumbTip.y) / 2 },
          nowMs
        );
        const palmPt = tr.palmFilter.filter(det.palm, nowMs);

        const pinchAng = Math.atan2(det.indexTip.y - det.thumbTip.y, det.indexTip.x - det.thumbTip.x);

        return { tr, det, allowed, POINT_ACTIVE, PALM_ACTIVE, pinchOn, pinchOff, pointPt, pinchPt, palmPt, pinchAng };
      });

      for (const x of act) {
        const tr = x.tr;
        if (!x.allowed) {
          tr.isPinching = false;
          tr.isOpenPalm = false;
          tr.isPointing = false;
          tr.drawStrokeId = "";
          tr.selectedStrokeId = "";
          continue;
        }

        if (!tr.isPinching && x.pinchOn) {
          tr.isPinching = true;
          tr.lastPinchPt = x.pinchPt;

          const picked = pickStroke(x.pinchPt);
          if (picked) tr.selectedStrokeId = picked;

          tr.pinchStartScalePx = x.det.scalePx;
          tr.pinchStartAng = x.pinchAng;
          tr.lastPinchAng = 0;
        } else if (tr.isPinching && x.pinchOff) {
          tr.isPinching = false;
        }
      }

      const pinchingAllowed = act.filter((x) => x.allowed && x.tr.isPinching);
      if (pinchingAllowed.length >= 2) {
        const a = pinchingAllowed[0];
        const b = pinchingAllowed[1];

        const pA = a.pinchPt;
        const pB = b.pinchPt;
        const mid: Pt = { x: (pA.x + pB.x) / 2, y: (pA.y + pB.y) / 2 };

        const d = Math.max(10, dist(pA, pB));
        const ang = Math.atan2(pB.y - pA.y, pB.x - pA.x);

        const tp = twoPinchRef.current;

        if (!tp.active) {
          tp.active = true;

          const sameSelected =
            a.tr.selectedStrokeId && a.tr.selectedStrokeId === b.tr.selectedStrokeId ? a.tr.selectedStrokeId : "";

          tp.strokeId = sameSelected || pickStroke(mid) || strokesRef.current.at(-1)?.id || "";
          tp.baseDist = d;
          tp.baseAng = ang;
          tp.lastK = 1;
          tp.lastAng = 0;
          tp.lastMid = mid;
        } else if (tp.strokeId) {
          const dx = mid.x - tp.lastMid.x;
          const dy = mid.y - tp.lastMid.y;
          tp.lastMid = mid;
          if (Math.abs(dx) + Math.abs(dy) > 0.2) translateStroke(tp.strokeId, dx, dy);

          const k = clamp(d / Math.max(10, tp.baseDist), 0.35, 3.0);
          const relK = k / Math.max(1e-6, tp.lastK);
          tp.lastK = k;
          if (Math.abs(relK - 1) > 0.006) scaleStroke(tp.strokeId, relK);

          const dAng = normAngleDelta(ang - tp.baseAng);
          const relAng = normAngleDelta(dAng - tp.lastAng);
          tp.lastAng = dAng;
          if (Math.abs(relAng) > 0.012) rotateStroke(tp.strokeId, relAng);
        }

        return;
      } else {
        twoPinchRef.current.active = false;
        twoPinchRef.current.strokeId = "";
      }

      for (const x of act) {
        const tr = x.tr;
        const det = x.det;

        if (!x.allowed) continue;

        if (!tr.isOpenPalm && x.PALM_ACTIVE) {
          tr.isOpenPalm = true;
          const picked = pickStroke(x.palmPt);
          if (picked) tr.selectedStrokeId = picked;
        } else if (tr.isOpenPalm && !x.PALM_ACTIVE) {
          tr.isOpenPalm = false;
        } else if (tr.isOpenPalm && tr.selectedStrokeId) {
          const ang = Math.atan2(det.indexMcp.y - det.wrist.y, det.indexMcp.x - det.wrist.x);
          const prev = tr.lastPinchAng;
          tr.lastPinchAng = ang;
          if (prev !== 0) {
            const dAng = normAngleDelta(ang - prev);
            if (Math.abs(dAng) > 0.02) rotateStroke(tr.selectedStrokeId, dAng);
          }
        }

        if (tr.isPinching && tr.selectedStrokeId) {
          const dx = x.pinchPt.x - tr.lastPinchPt.x;
          const dy = x.pinchPt.y - tr.lastPinchPt.y;
          tr.lastPinchPt = x.pinchPt;
          if (Math.abs(dx) + Math.abs(dy) > 0.2) translateStroke(tr.selectedStrokeId, dx, dy);

          const dAng = normAngleDelta(x.pinchAng - tr.pinchStartAng);
          const relAng = normAngleDelta(dAng - tr.lastPinchAng);
          tr.lastPinchAng = dAng;
          if (Math.abs(relAng) > 0.012) rotateStroke(tr.selectedStrokeId, relAng);

          const rel = clamp(det.scalePx / Math.max(1, tr.pinchStartScalePx), 0.92, 1.08);
          tr.pinchStartScalePx = det.scalePx;
          if (Math.abs(rel - 1) > 0.01) scaleStroke(tr.selectedStrokeId, rel);
        }

        if (!tr.isPointing && x.POINT_ACTIVE && !tr.isPinching) {
          tr.isPointing = true;
          const isAdmin = isAdminGroup(tr);
          const color = effectiveColorForHand(isAdmin);
          maybeUnlockGuestsByAdminBlackStrokeStart(tr, color);

          const adminS = adminScaleRef.current || det.scalePx;
          const handScale = clamp(det.scalePx / Math.max(1, adminS), 0.8, 1.35);
          const thickness = clamp(settingsRef.current.baseThickness * handScale, 1, 40);

          tr.drawStrokeId = startStroke(color, x.pointPt, nowMs, thickness);
        } else if (tr.isPointing && (!x.POINT_ACTIVE || tr.isPinching)) {
          tr.isPointing = false;
          tr.drawStrokeId = "";
        } else if (tr.isPointing && tr.drawStrokeId) {
          const adminS = adminScaleRef.current || det.scalePx;
          const handScale = clamp(det.scalePx / Math.max(1, adminS), 0.8, 1.35);
          const thickness = clamp(settingsRef.current.baseThickness * handScale + det.pinchStrength * 2.2, 1, 40);
          addStrokePoint(tr.drawStrokeId, x.pointPt, nowMs, thickness);
        }
      }
    },
    [
      addStrokePoint,
      effectiveColorForHand,
      isAdminGroup,
      isAllowedToControl,
      maybeUnlockGuestsByAdminBlackStrokeStart,
      pickStroke,
      rotateStroke,
      scaleStroke,
      startStroke,
      translateStroke,
    ]
  );

  /* ------------------------------ overlay render ------------------------------ */
  const renderOverlay = useCallback(() => {
    const ov = overlayRef.current;
    const canvas = drawRef.current;
    if (!ov || !canvas) return;

    const ctx = ov.getContext("2d")!;
    ctx.clearRect(0, 0, ov.width, ov.height);
    const drawUtils = new DrawingUtils(ctx);

    const tracks = tracksRef.current;

    // Drawing skeleton if enabled
    if (settingsRef.current.showLandmarks) {
      for (const tr of tracks) {
        if (!tr.det.rawLandmarks?.length) continue;
        const admin = isAdminGroup(tr);
        // Map landmarks back to 0.0-1.0 for DrawingUtils
        const normalizedLandmarks = tr.det.rawLandmarks.map(p => ({ 
          x: p.x / canvas.width, 
          y: p.y / canvas.height,
          z: 0,
          visibility: 1
        })) as any;
        
        drawUtils.drawConnectors(normalizedLandmarks, GestureRecognizer.HAND_CONNECTIONS, {
          color: admin ? "#69f0ae" : "#ffffff",
          lineWidth: 2
        });
        drawUtils.drawLandmarks(normalizedLandmarks, {
          color: admin ? "#69f0ae" : "#ffffff",
          radius: 3
        });
      }
    }

    for (const tr of tracks) {
      const allowed = isAllowedToControl(tr);
      const admin = isAdminGroup(tr);
      const color = admin ? "rgba(105,240,174,0.45)" : allowed ? "rgba(255,204,102,0.4)" : "rgba(255,85,102,0.35)";

      ctx.save();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(tr.det.indexTip.x, tr.det.indexTip.y, 12, 0, Math.PI * 2);
      ctx.fill();

      // Undo mirroring for text
      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-ov.width, 0);
      
      const label = admin ? "ADMIN" : allowed ? "GUEST" : "BLOCKED";
      const gesture = tr.det.gesture && tr.det.gesture !== "None" ? ` [${tr.det.gesture}]` : "";
      const conf = tr.det.gestureConfidence ? ` ${(tr.det.gestureConfidence * 100).toFixed(0)}%` : "";
      
      ctx.font = `bold 14px ${fontStack}`;
      ctx.fillStyle = admin ? "rgba(105,240,174,1.0)" : allowed ? "rgba(255,204,102,1.0)" : "rgba(255,85,102,1.0)";
      
      const tx = ov.width - tr.det.indexTip.x + 16;
      const ty = tr.det.indexTip.y - 12;
      ctx.fillText(label + gesture + conf, tx, ty);
      ctx.restore();
      ctx.restore();

      if (tr.selectedStrokeId) {
        const s = strokesRef.current.find((x) => x.id === tr.selectedStrokeId);
        if (s) {
          const bb = padBBox(s.bbox, 12);
          ctx.save();
          ctx.setLineDash([8, 6]);
          ctx.lineWidth = 2;
          ctx.strokeStyle = admin ? "rgba(105,240,174,0.9)" : "rgba(255,204,102,0.9)";
          ctx.strokeRect(bb.minX, bb.minY, bb.maxX - bb.minX, bb.maxY - bb.minY);
          ctx.restore();
        }
      }
    }

    ctx.save();
    ctx.scale(-1, 1);
    ctx.translate(-ov.width, 0);
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.fillRect(ov.width - 700, ov.height - 46, 688, 32);
    ctx.fillStyle = "rgba(255,255,255,0.95)";
    ctx.font = `12px ${fontStack}`;
    const g = guestsUnlocked() ? "ON" : "OFF";
    ctx.fillText(`Hands ${tracks.length} • Guest ${g} • Voice ${voiceRef.current?.name ?? "default"}`, ov.width - 690, ov.height - 26);
    ctx.restore();
  }, [fontStack, guestsUnlocked, isAdminGroup, isAllowedToControl]);

  useEffect(() => {
    let mounted = true;

    const onResize = () => resizeCanvasesToVideo();
    window.addEventListener("resize", onResize);

    (async () => {
      setLoadingStep("1/4 Camera…");
      try {
        await startCamera();
      } catch (e: any) {
        setLoadingStep(`Cam ERROR: ${e.name || "Access"}`);
        return;
      }

      setLoadingStep("2/4 Assets…");
      const loadWithTimeout = async <T,>(promise: Promise<T>, timeoutMs: number, label: string): Promise<T> => {
        return Promise.race([
          promise,
          new Promise<T>((_, reject) => setTimeout(() => reject(new Error(`${label} timeout`)), timeoutMs))
        ]);
      };

      try {
        const vision = await loadWithTimeout(
          FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm"),
          15000,
          "WASM"
        );

        const modelPath = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";

        const tryCreate = async (delegate: "GPU" | "CPU") => {
          setLoadingStep(`3/4 Init ${delegate}…`);
          return await loadWithTimeout(GestureRecognizer.createFromOptions(vision, {
            baseOptions: { modelAssetPath: modelPath, delegate },
            runningMode: "VIDEO",
            numHands: 2, // explicit for robustness
            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
          }), 15000, `Model-${delegate}`);
        };

        let gr: GestureRecognizer;
        try {
          gr = await tryCreate("GPU");
        } catch (e) {
          console.warn("GPU init failed, falling back to CPU...", e);
          gr = await tryCreate("CPU");
        }

        if (!mounted) {
          gr.close();
          return;
        }

        setLandmarker(gr);
        setReady(true);
        setLoadingStep("Ready");
        setVoiceHint("Running. Point → draw. Fist → grab. Palm → select.");
      } catch (e: any) {
        console.error("Boot failed:", e);
        setLoadingStep(`Fatal: ${e.message?.slice(0, 20)}`);
        speak("Failed to start.");
      }
    })().catch((e) => {
      console.error(e);
      setLoadingStep("FATAL");
    });

    return () => {
      mounted = false;
      window.removeEventListener("resize", onResize);
      const stream = videoRef.current?.srcObject as MediaStream | null;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [resizeCanvasesToVideo, speak, startCamera]);

  /* ------------------------------ inference loop ------------------------------ */
  useEffect(() => {
    if (!ready || !landmarker) return;

    let alive = true;

    const v = videoRef.current;
    const infer = inferCanvasRef.current;
    const lm = landmarker;
    if (!v || !infer || !lm) return;

    const ictx = infer.getContext("2d")!;
    ictx.imageSmoothingEnabled = false;

    let lastInferMs = 0;
    let lastHudMs = 0;
    let emaInfer = 0;

    const schedule = (cb: (t: number) => void) => {
      const vAny = v as any;
      if (typeof vAny.requestVideoFrameCallback === "function") vAny.requestVideoFrameCallback(cb);
      else requestAnimationFrame(cb);
    };

    const step = (now: number) => {
      if (!alive) return;

      const minDt = 1000 / Math.max(5, settingsRef.current.inferFpsCap);

      if (now - lastInferMs >= minDt && v.readyState >= 2) {
        lastInferMs = now;

        const sz = getInferSize(settingsRef.current.inferPreset);
        if (infer.width !== sz.w || infer.height !== sz.h) {
          infer.width = sz.w;
          infer.height = sz.h;
        }

        ictx.drawImage(v, 0, 0, infer.width, infer.height);

        const t0 = performance.now();
        const res = lm.recognizeForVideo(v, performance.now());
        const t1 = performance.now();

        const inferMs = t1 - t0;
        emaInfer = emaInfer === 0 ? inferMs : emaInfer * 0.85 + inferMs * 0.15;

        const landmarks = res.landmarks || [];
        const handedness = (res.handedness || (res as any).handednesses || []) as any[];
        const gestureResults = res.gestures || [];

        const draw = drawRef.current;
        if (draw) {
          const W = draw.width;
          const H = draw.height;

          const dets: HandDet[] = [];

          for (let i = 0; i < landmarks.length; i++) {
            const hand = landmarks[i];
            if (!hand?.length) continue;

            const hd0 = handedness[i]?.[0];
            const handName = (hd0?.categoryName ?? "Unknown") as "Left" | "Right" | "Unknown";

            const wrist = { x: hand[IDX.wrist].x * W, y: hand[IDX.wrist].y * H };
            const indexMcp = { x: hand[IDX.indexMcp].x * W, y: hand[IDX.indexMcp].y * H };
            const middleMcp = { x: hand[IDX.middleMcp].x * W, y: hand[IDX.middleMcp].y * H };

            const thumbTip = { x: hand[IDX.thumbTip].x * W, y: hand[IDX.thumbTip].y * H };
            const indexTip = { x: hand[IDX.indexTip].x * W, y: hand[IDX.indexTip].y * H };
            const indexPip = { x: hand[IDX.indexPip].x * W, y: hand[IDX.indexPip].y * H };

            const middleTip = { x: hand[IDX.middleTip].x * W, y: hand[IDX.middleTip].y * H };
            const middlePip = { x: hand[IDX.middlePip].x * W, y: hand[IDX.middlePip].y * H };

            const ringTip = { x: hand[IDX.ringTip].x * W, y: hand[IDX.ringTip].y * H };
            const ringPip = { x: hand[IDX.ringPip].x * W, y: hand[IDX.ringPip].y * H };

            const pinkyTip = { x: hand[IDX.pinkyTip].x * W, y: hand[IDX.pinkyTip].y * H };
            const pinkyPip = { x: hand[IDX.pinkyPip].x * W, y: hand[IDX.pinkyPip].y * H };

            const palm = {
              x: ((hand[IDX.wrist].x + hand[IDX.indexMcp].x + hand[IDX.middleMcp].x + hand[IDX.ringMcp].x + hand[IDX.pinkyMcp].x) / 5) * W,
              y: ((hand[IDX.wrist].y + hand[IDX.indexMcp].y + hand[IDX.middleMcp].y + hand[IDX.ringMcp].y + hand[IDX.pinkyMcp].y) / 5) * H,
            };

            const scalePx = Math.max(60, dist(wrist, middleMcp));
            const pinchDist = dist(indexTip, thumbTip);
            const pinchStrength = clamp(1 - pinchDist / (scalePx * 0.55), 0, 1);

            const rawLandmarks = hand.map((pixelLm: any) => ({ x: pixelLm.x * W, y: pixelLm.y * H }));

            const gFull = gestureResults[i]?.[0];
            const gestureName = gFull?.categoryName ?? "None";
            const gestureScore = gFull?.score ?? 0;

            const base = {
              frameIndex: i,
              handedness: handName,
              wrist,
              palm,
              indexTip,
              indexPip,
              thumbTip,
              indexMcp,
              middleMcp,
              middleTip,
              ringTip,
              pinkyTip,
              middlePip,
              ringPip,
              pinkyPip,
              scalePx,
              pinchStrength,
            } as Omit<HandDet, "rawPoint" | "rawPalm" | "rawLandmarks" | "gesture" | "gestureConfidence">;

            const g = computeGestureBooleans(base);
            
            // Override manual booleans with robust MediaPipe gestures
            const isPointing = gestureName === "Pointing_Up";
            const isPalm = gestureName === "Open_Palm";
            const isGrab = gestureName === "Closed_Fist" || gestureName === "Thumb_Up";

            dets.push({ 
              ...base, 
              rawPoint: isPointing || g.rawPoint, 
              rawPalm: isPalm || g.rawPalm, 
              rawGrab: isGrab,
              rawLandmarks,
              gesture: gestureName,
              gestureConfidence: gestureScore
            });
          }

          assignTracks(dets, now);
          updateAdminScaleFromTracks();
          processInteractions(now);
        }

        if (now - lastHudMs > 250) {
          lastHudMs = now;
          setHud((h) => ({
            ...h,
            inferMs: emaInfer,
            strokes: strokesRef.current.length,
            hands: tracksRef.current.length,
            guestUnlocked: guestsUnlocked(),
          }));
        }
      }

      schedule((t) => step(t));
    };

    schedule((t) => step(t));

    return () => {
      alive = false;
    };
  }, [assignTracks, guestsUnlocked, processInteractions, ready, landmarker, updateAdminScaleFromTracks]);

  /* ------------------------------ render loop + FPS ------------------------------ */
  useEffect(() => {
    let alive = true;
    let last = performance.now();
    let ema = 0;
    let lastHudTs = 0;

    const loop = (now: number) => {
      if (!alive) return;

      const dt = Math.max(1, now - last);
      const inst = 1000 / dt;
      ema = ema === 0 ? inst : ema * 0.9 + inst * 0.1;
      last = now;

      if (needsFullRedrawRef.current) {
        needsFullRedrawRef.current = false;
        redrawAll();
      }
      renderOverlay();

      if (now - lastHudTs > 250) {
        lastHudTs = now;
        setHud((h) => ({ ...h, fps: ema }));
      }

      requestAnimationFrame(loop);
    };

    requestAnimationFrame(loop);
    return () => {
      alive = false;
    };
  }, [redrawAll, renderOverlay]);

  /* ------------------------------ voice toggle ------------------------------ */
  useEffect(() => {
    if (!voiceOn) {
      try { speechRecRef.current?.stop(); } catch {}
      setVoiceHint("Mic off.");
      return;
    }

    if (!speechRecRef.current) speechRecRef.current = setupSpeechRecognition();
    const rec = speechRecRef.current;
    if (!rec) return;

    try {
      rec.start();
    } catch (e: any) {
      if (e.name !== "InvalidStateError") {
        console.warn("[Voice] Start error:", e);
      }
    }
  }, [setupSpeechRecognition, voiceOn]);

  /* ------------------------------ UI state -> settingsRef ------------------------------ */
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [glow, setGlow] = useState(true);
  const [brushColor, setBrushColor] = useState("#69F0AE");
  const [baseThickness, setBaseThickness] = useState(6);
  const [inferPreset, setInferPreset] = useState<InferencePreset>("FAST");
  const [inferFpsCap, setInferFpsCap] = useState(24);
  const [maxHands, setMaxHands] = useState(4);

  useEffect(() => {
    settingsRef.current.showLandmarks = showLandmarks;
  }, [showLandmarks]);
  useEffect(() => {
    settingsRef.current.glow = glow;
    needsFullRedrawRef.current = true;
  }, [glow]);
  useEffect(() => {
    settingsRef.current.brushColor = brushColor;
  }, [brushColor]);
  useEffect(() => {
    settingsRef.current.baseThickness = baseThickness;
  }, [baseThickness]);
  useEffect(() => {
    settingsRef.current.inferPreset = inferPreset;
    const infer = inferCanvasRef.current;
    if (infer) {
      const { w, h } = getInferSize(inferPreset);
      infer.width = w;
      infer.height = h;
    }
  }, [inferPreset]);
  useEffect(() => {
    settingsRef.current.inferFpsCap = inferFpsCap;
  }, [inferFpsCap]);
  useEffect(() => {
    settingsRef.current.maxHands = maxHands;
    landmarker?.setOptions?.({ numHands: maxHands }).catch?.(() => {});
  }, [maxHands]);

  /* ------------------------------ UI styles ------------------------------ */
  const ui = {
    card: {
      background: theme.panel,
      border: `1px solid ${theme.border}`,
      borderRadius: 16,
      padding: 12,
      backdropFilter: "blur(10px)",
    } as CSSProperties,
    btn: (primary?: boolean) =>
      ({
        padding: "10px 12px",
        borderRadius: 12,
        border: `1px solid ${primary ? theme.border2 : theme.border}`,
        background: primary ? "rgba(105,240,174,0.18)" : "rgba(0,0,0,0.25)",
        color: theme.fg,
        cursor: "pointer",
        fontWeight: 800,
        fontFamily: fontStack,
      }) as CSSProperties,
    label: {
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      gap: 10,
      color: theme.muted,
      fontSize: 12,
    } as CSSProperties,
  };

  const onQuickShape = (shape: "circle" | "square" | "triangle" | "heart" | "star" | "arrow") => {
    drawShapeAtCenter(shape, "small", settingsRef.current.brushColor);
  };

  return (
    <div style={{ background: theme.bg, minHeight: "100vh", color: theme.fg, padding: 16, fontFamily: fontStack }}>
      <div style={{ maxWidth: 1200, margin: "0 auto", display: "grid", gridTemplateColumns: "320px 1fr", gap: 14 }}>
        {/* Left Panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={ui.card}>
            <div style={{ fontSize: 18, fontWeight: 900 }}>Neon Studio</div>
            <div style={{ fontSize: 12, color: theme.muted, marginTop: 6 }}>
              Point → draw • Pinch → grab/move • While pinching: twist to rotate, move closer/farther to scale • Two pinches → move + scale + rotate
            </div>
            <div style={{ fontSize: 12, color: theme.muted, marginTop: 6 }}>
              Admin locks control; guests unlock only when admin draws with <b>black</b> (guests forced black).
            </div>
          </div>

          <div style={ui.card}>
            <div style={{ fontSize: 13, fontWeight: 900, marginBottom: 10 }}>Tools</div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <button style={ui.btn()} onClick={undo}>Undo</button>
              <button style={ui.btn()} onClick={redo}>Redo</button>
              <button style={ui.btn()} onClick={clearAll}>Clear</button>
              <button style={ui.btn(true)} onClick={() => void downloadPng()}>Export PNG</button>
              <button style={ui.btn(true)} onClick={() => void sharePng()}>Share</button>
            </div>

            <div style={{ height: 12 }} />

            <div style={ui.label}>
              <span>Brush color</span>
              <input
                type="color"
                value={brushColor.startsWith("#") ? brushColor : "#69f0ae"}
                onChange={(e) => setBrushColor(e.target.value)}
                style={{ width: 44, height: 28, background: "transparent", border: "none" }}
              />
            </div>

            <div style={{ height: 10 }} />

            <div style={ui.label}>
              <span>Thickness</span>
              <input
                type="range"
                min={1}
                max={40}
                step={0.5}
                value={baseThickness}
                onChange={(e) => setBaseThickness(parseFloat(e.target.value))}
              />
              <span style={{ width: 48, textAlign: "right" }}>{baseThickness.toFixed(1)}</span>
            </div>

            <div style={{ height: 10 }} />

            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <button style={ui.btn()} onClick={() => setGlow((v) => !v)}>{glow ? "Glow: On" : "Glow: Off"}</button>
              <button style={ui.btn()} onClick={() => setShowLandmarks((v) => !v)}>{showLandmarks ? "Hide Landmarks" : "Show Landmarks"}</button>
            </div>

            <div style={{ height: 12 }} />

            <div style={{ fontSize: 13, fontWeight: 900, marginBottom: 10 }}>Quick Shapes</div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <button style={ui.btn()} onClick={() => onQuickShape("circle")}>Circle</button>
              <button style={ui.btn()} onClick={() => onQuickShape("square")}>Square</button>
              <button style={ui.btn()} onClick={() => onQuickShape("triangle")}>Triangle</button>
              <button style={ui.btn()} onClick={() => onQuickShape("heart")}>Heart</button>
              <button style={ui.btn()} onClick={() => onQuickShape("star")}>Star</button>
              <button style={ui.btn()} onClick={() => onQuickShape("arrow")}>Arrow</button>
            </div>
          </div>

          <div style={ui.card}>
            <div style={{ fontSize: 13, fontWeight: 900, marginBottom: 10 }}>Social Share (Copy & Open)</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <button 
                style={{ ...ui.btn(), background: "rgba(37,211,102,0.15)", border: "1px solid rgba(37,211,102,0.3)" }} 
                onClick={() => void sharePng("whatsapp")}
              >
                WhatsApp
              </button>
              <button 
                style={{ ...ui.btn(), background: "rgba(225,48,108,0.15)", border: "1px solid rgba(225,48,108,0.3)" }} 
                onClick={() => void sharePng("instagram")}
              >
                Instagram
              </button>
              <button 
                style={{ ...ui.btn(), background: "rgba(24,119,242,0.15)", border: "1px solid rgba(24,119,242,0.3)" }} 
                onClick={() => void sharePng("facebook")}
              >
                Facebook
              </button>
              <button 
                style={{ ...ui.btn(), background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.2)" }} 
                onClick={() => void sharePng("x")}
              >
                X / Twitter
              </button>
            </div>
            <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
              <button style={{ ...ui.btn(), flex: 1 }} onClick={() => void copyToClipboard()}>Copy Image</button>
              <button style={{ ...ui.btn(), flex: 1 }} onClick={() => void sharePng()}>Other...</button>
            </div>
            <div style={{ marginTop: 8, fontSize: 11, color: theme.muted }}>
              Automatically copies image so you can just <b>Paste</b> in the app.
            </div>
          </div>

          <div style={ui.card}>
            <div style={{ fontSize: 13, fontWeight: 900, marginBottom: 10 }}>Performance</div>

            <div style={ui.label}>
              <span>Inference preset</span>
              <select
                value={inferPreset}
                onChange={(e) => setInferPreset(e.target.value as InferencePreset)}
                style={{
                  background: "rgba(0,0,0,0.35)",
                  border: `1px solid ${theme.border}`,
                  borderRadius: 10,
                  color: theme.fg,
                  padding: "8px 10px",
                  fontFamily: fontStack,
                }}
              >
                <option value="FAST">FAST (best speed)</option>
                <option value="BALANCED">BALANCED</option>
                <option value="HIGH">HIGH (best accuracy)</option>
              </select>
            </div>

            <div style={{ height: 10 }} />

            <div style={ui.label}>
              <span>Inference FPS cap</span>
              <input type="range" min={10} max={30} step={1} value={inferFpsCap} onChange={(e) => setInferFpsCap(parseInt(e.target.value, 10))} />
              <span style={{ width: 32, textAlign: "right" }}>{inferFpsCap}</span>
            </div>

            <div style={{ height: 10 }} />

            <div style={ui.label}>
              <span>Max hands</span>
              <input type="range" min={1} max={4} step={1} value={maxHands} onChange={(e) => setMaxHands(parseInt(e.target.value, 10))} />
              <span style={{ width: 32, textAlign: "right" }}>{maxHands}</span>
            </div>

            <div style={{ marginTop: 10, fontSize: 12, color: theme.muted }}>
              Tip: Low-end Android → FAST + 18–24 FPS cap.
            </div>
          </div>

          <div style={ui.card}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
              <div style={{ fontSize: 13, fontWeight: 900 }}>Voice</div>
              <div style={{ display: "flex", gap: 6 }}>
                <button style={ui.btn()} onClick={resetVoice}>Reset</button>
                <button style={ui.btn(voiceOn)} onClick={() => setVoiceOn((v) => !v)}>{voiceOn ? "Mic: On" : "Mic: Off"}</button>
              </div>
            </div>

            <div style={{ marginTop: 10, fontSize: 12, color: theme.muted }}>{voiceHint}</div>
            <div style={{ marginTop: 8, fontSize: 12, color: theme.muted }}>
              Commands: “change color to red”, “glow off”, “thickness 12”, “draw small star”, “unlock guests”, “export”.
            </div>
          </div>

          <div style={ui.card}>
            <div style={{ fontSize: 12, color: theme.muted }}>
              Status: {ready ? "Running" : loadingStep} • FPS {hud.fps.toFixed(1)} • Infer {hud.inferMs.toFixed(1)}ms • Strokes {hud.strokes} • Hands {hud.hands} • Guest {hud.guestUnlocked ? "ON" : "OFF"}
            </div>
          </div>
        </div>

        {/* Canvas Area */}
        <div
          style={{
            position: "relative",
            border: `1px solid ${theme.border}`,
            borderRadius: 18,
            overflow: "hidden",
            aspectRatio: "16/9",
            background: "rgba(255,255,255,0.03)",
          }}
        >
          <video
            ref={videoRef}
            muted
            playsInline
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              objectFit: "cover",
              transform: "scaleX(-1)",
            }}
          />

          <canvas
            ref={drawRef}
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              pointerEvents: "none",
              transform: "scaleX(-1)",
            }}
          />

          <canvas
            ref={overlayRef}
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              pointerEvents: "none",
              transform: "scaleX(-1)",
            }}
          />

          <canvas ref={inferCanvasRef} style={{ display: "none" }} />

          <div
            style={{
              position: "absolute",
              inset: 0,
              pointerEvents: "none",
              boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.05), inset 0 0 60px rgba(0,0,0,0.35)",
            }}
          />
        </div>
      </div>
    </div>
  );
}