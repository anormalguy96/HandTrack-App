import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { CSSProperties } from "react";
import {
  FilesetResolver,
  GestureRecognizer,
  FaceLandmarker, // [NEW] Added for AR
  HandLandmarkerResult,
  GestureRecognizerResult,
  FaceLandmarkerResult,
} from "@mediapipe/tasks-vision";
import "./App.css";

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

// Hand connections (avoid DrawingUtils allocations & normalize mapping)
const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
];

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
  return {
    minX: bb.minX - pad,
    minY: bb.minY - pad,
    maxX: bb.maxX + pad,
    maxY: bb.maxY + pad,
  };
}

function isMobile() {
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent,
  );
}
function bboxContains(bb: BBox, p: Pt) {
  return p.x >= bb.minX && p.x <= bb.maxX && p.y >= bb.minY && p.y <= bb.maxY;
}
function computeBBox(pts: { x: number; y: number }[]): BBox {
  if (!pts.length) return emptyBBox();
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  for (const p of pts) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }
  return { minX, minY, maxX, maxY };
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
  constructor(
    private minCutoff = 0.9,
    private beta = 0.02,
    private dCutoff = 1.0,
  ) {}
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
  handIndex: number;
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
  rawLandmarks: Array<{ x: number; y: number; z?: number }>; // normalized landmarks from MediaPipe
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
function computeGestureBooleans(
  det: Omit<HandDet, "rawPoint" | "rawPalm" | "rawGrab" | "rawLandmarks">,
): Pick<HandDet, "rawPoint" | "rawPalm"> {
  const margin = clamp(det.scalePx * 0.06, 6, 18);
  const indexExt = isFingerExtended(
    det.wrist,
    det.indexTip,
    det.indexPip,
    margin,
  );
  const middleExt = isFingerExtended(
    det.wrist,
    det.middleTip,
    det.middlePip,
    margin,
  );
  const ringExt = isFingerExtended(det.wrist, det.ringTip, det.ringPip, margin);
  const pinkyExt = isFingerExtended(
    det.wrist,
    det.pinkyTip,
    det.pinkyPip,
    margin,
  );

  const rawPoint = indexExt && !middleExt && !ringExt && !pinkyExt;
  const rawPalm =
    indexExt && middleExt && ringExt && pinkyExt && det.pinchStrength < 0.2;

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

  const ones: Record<string, number> = {
    zero: 0,
    one: 1,
    two: 2,
    three: 3,
    four: 4,
    five: 5,
    six: 6,
    seven: 7,
    eight: 8,
    nine: 9,
  };
  const teens: Record<string, number> = {
    ten: 10,
    eleven: 11,
    twelve: 12,
    thirteen: 13,
    fourteen: 14,
    fifteen: 15,
    sixteen: 16,
    seventeen: 17,
    eighteen: 18,
    nineteen: 19,
  };
  const tens: Record<string, number> = {
    twenty: 20,
    thirty: 30,
    forty: 40,
    fifty: 50,
    sixty: 60,
    seventy: 70,
    eighty: 80,
    ninety: 90,
  };

  function parseIntPhrase(tokens: string[]) {
    let i = 0;
    let val = 0;
    while (i < tokens.length) {
      const t = tokens[i];
      if (t in ones) {
        val += ones[t];
        i++;
      } else if (t in teens) {
        val += teens[t];
        i++;
      } else if (t in tens) {
        val += tens[t];
        if (i + 1 < tokens.length && tokens[i + 1] in ones) {
          val += ones[tokens[i + 1]];
          i += 2;
        } else i++;
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

const fontStack =
  'Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji"';

// SVG Icons
const Icons = {
  Undo: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 7v6h6" />
      <path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13" />
    </svg>
  ),
  Redo: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 7v6h-6" />
      <path d="M3 17a9 9 0 0 1 9-9 9 9 0 0 1 6 2.3L21 13" />
    </svg>
  ),
  Clear: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </svg>
  ),
  Export: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  ),
  Share: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="18" cy="5" r="3" />
      <circle cx="6" cy="12" r="3" />
      <circle cx="18" cy="19" r="3" />
      <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" />
      <line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
    </svg>
  ),
  Glow: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2" />
      <path d="M12 20v2" />
      <path d="M4.93 4.93l1.41 1.41" />
      <path d="M17.66 17.66l1.41 1.41" />
      <path d="M2 12h2" />
      <path d="M20 12h2" />
      <path d="M6.34 17.66l-1.41 1.41" />
      <path d="M19.07 4.93l-1.41 1.41" />
    </svg>
  ),
  Landmarks: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="5" r="1" />
      <circle cx="12" cy="19" r="1" />
      <circle cx="5" cy="12" r="1" />
      <circle cx="19" cy="12" r="1" />
      <path d="M5 12h14" />
      <path d="M12 5v14" />
    </svg>
  ),
  MicOn: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
  ),
  MicOff: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="1" y1="1" x2="23" y2="23" />
      <path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6" />
      <path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23" />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
  ),
  Circle: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
    </svg>
  ),
  Square: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
    </svg>
  ),
  Triangle: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    </svg>
  ),
  Heart: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
    </svg>
  ),
  Star: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
    </svg>
  ),
  Save: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
      <polyline points="17 21 17 13 7 13 7 21" />
      <polyline points="7 3 7 8 15 8" />
    </svg>
  ),
  Copy: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  ),
  Eraser: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M20 20H7L3 16c-1-1-1-2.5 0-3.5l9.5-9.5c1-1 2.5-1 3.5 0l5.5 5.5c1 1 1 2.5 0 3.5L13 20.5" />
      <line x1="7" y1="20" x2="20" y2="20" />
    </svg>
  ),
  Arrow: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="12" y1="19" x2="12" y2="5" />
      <polyline points="5 12 12 5 19 12" />
    </svg>
  ),
  WhatsApp: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
    </svg>
  ),
  Instagram: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="2" y="2" width="20" height="20" rx="5" ry="5" />
      <path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z" />
      <line x1="17.5" y1="6.5" x2="17.51" y2="6.5" />
    </svg>
  ),
  Facebook: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M18 2h-3a5 5 0 0 0-5 5v3H7v4h3v8h4v-8h3l1-4h-4V7a1 1 0 0 1 1-1h3z" />
    </svg>
  ),
  Twitter: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M23 3a10.9 10.9 0 0 1-3.14 1.53 4.48 4.48 0 0 0-7.86 3v1A10.66 10.66 0 0 1 3 4s-4 9 5 13a11.64 11.64 0 0 1-7 2c9 5 20 0 20-11.5a4.5 4.5 0 0 0-.08-.83A7.72 7.72 0 0 0 23 3z" />
    </svg>
  ),
  Cameraswitch: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M7 21l-4-4 4-4" />
      <path d="M3 17h18" />
      <path d="M17 3l4 4-4 4" />
      <path d="M21 7H3" />
    </svg>
  ),
  Settings: () => (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  ),
};

function getMapScale(
  videoW: number,
  videoH: number,
  canvasW: number,
  canvasH: number,
  objectFit: "fill" | "cover" = "fill",
) {
  if (objectFit === "fill") {
    return {
      scaleX: canvasW / videoW,
      scaleY: canvasH / videoH,
      offsetX: 0,
      offsetY: 0,
    };
  }
  const scale = Math.max(canvasW / videoW, canvasH / videoH);
  const visualW = videoW * scale;
  const visualH = videoH * scale;
  const offsetX = (visualW - canvasW) / 2;
  const offsetY = (visualH - canvasH) / 2;
  return { scaleX: scale, scaleY: scale, offsetX, offsetY };
}

/* ------------------------------- Main Component --------------------------- */
export default function App() {
  /* ------------------------------- DOM refs ------------------------------- */
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const drawRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const inferCanvasRef = useRef<HTMLCanvasElement | null>(null);

  /* -------------------------- MediaPipe landmarker ------------------------- */
  /* -------------------------- MediaPipe landmarker ------------------------- */
  const [landmarker, setLandmarker] = useState<GestureRecognizer | null>(null);
  const [faceLandmarker, setFaceLandmarker] = useState<FaceLandmarker | null>(
    null,
  ); // [NEW] Ref for face
  const [arItems, setArItems] = useState<{ hat: boolean; glasses: boolean }>({
    hat: false,
    glasses: false,
  }); // [NEW] AR State

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
    [],
  );

  /* --------------------------- core runtime refs --------------------------- */
  const strokesRef = useRef<Stroke[]>([]);
  const redoRef = useRef<Stroke[]>([]);
  const tracksRef = useRef<Track[]>([]);
  const faceResultsRef = useRef<FaceLandmarkerResult | null>(null); // [NEW] Ref to bridge inference and render loops
  const needsFullRedrawRef = useRef(false);

  // Fast lookup for strokes
  const strokeByIdRef = useRef<Map<string, Stroke>>(new Map());

  // AR enabled flag without causing inference loop re-renders
  const arEnabledRef = useRef(false);
  useEffect(() => {
    arEnabledRef.current = !!(arItems.hat || arItems.glasses);
  }, [arItems]);

  // admin/guest control
  const adminScaleRef = useRef<number>(0);
  const guestUnlockUntilRef = useRef<number>(0);
  const adminTolerance = 0.22;

  // Export session tracking
  const lastSavedStrokeIdRef = useRef<string | null>(null);

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
    eraserMode: false, // New eraser mode (Type 2)

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
  const [hud, setHud] = useState({
    fps: 0,
    inferMs: 0,
    strokes: 0,
    hands: 0,
    guestUnlocked: false,
  });
  const [showInstructions, setShowInstructions] = useState(false);
  const [showMobileSettings, setShowMobileSettings] = useState(false); // [NEW] Toggle for mobile settings
  const [initialOverlay, setInitialOverlay] = useState(true);

  /* -------------------------------- voice -------------------------------- */
  const [voiceOn, setVoiceOn] = useState(true);
  const voiceOnRef = useRef(true);
  useEffect(() => {
    voiceOnRef.current = voiceOn;
  }, [voiceOn]);
  const [voiceHint, setVoiceHint] = useState(
    'Try: "Tensor change color to red", "Tensor glow off", "Tensor export".',
  );
  const [isSpeaking, setIsSpeaking] = useState(false);

  // New UI States
  const [brushColor, setBrushColor] = useState(settingsRef.current.brushColor);
  const [baseThickness, setBaseThickness] = useState(
    settingsRef.current.baseThickness,
  );
  const [glow, setGlow] = useState(settingsRef.current.glow);
  const [showLandmarks, setShowLandmarks] = useState(
    settingsRef.current.showLandmarks,
  );
  const [inferPreset, setInferPreset] = useState<InferencePreset>(
    settingsRef.current.inferPreset,
  );
  const [inferFpsCap, setInferFpsCap] = useState(
    settingsRef.current.inferFpsCap,
  );
  const [maxHands, setMaxHands] = useState(settingsRef.current.maxHands);
  const [eraserMode, setEraserMode] = useState(false); // Type 2 eraser UI state
  const [facingMode, setFacingMode] = useState<"user" | "environment">("user");
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [preferIntegrated, setPreferIntegrated] = useState(true);
  const [videoAspect, setVideoAspect] = useState(16 / 9);

  // Auto-hide initial overlay
  useEffect(() => {
    const timer = setTimeout(() => setInitialOverlay(false), 3500);
    return () => clearTimeout(timer);
  }, []);

  /* ------------------------------- VAD / Voice ------------------------------ */
  const vadContextRef = useRef<AudioContext | null>(null);
  const vadAnalyserRef = useRef<AnalyserNode | null>(null);
  const vadSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const vadStreamRef = useRef<MediaStream | null>(null);
  const vadIntervalRef = useRef<number>(0);

  const speechRecRef = useRef<any>(null);
  const voiceRef = useRef<SpeechSynthesisVoice | null>(null);
  // Guard to prevent self-triggering
  const ttsGuardRef = useRef({
    speaking: false,
    lastSpoken: "",
    ignoreUntil: 0,
    suppressAutoRestart: false,
    lastCmd: { text: "", at: 0 },
  });

  const handleVoiceCommandRef = useRef<((t: string) => void) | null>(null);

  // Initialize VAD
  useEffect(() => {
    // Only run VAD if voice is locally enabled
    if (!voiceOn) return;

    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: true,
        });
        vadStreamRef.current = stream;

        const ctx = new (
          window.AudioContext || (window as any).webkitAudioContext
        )();
        vadContextRef.current = ctx;

        const source = ctx.createMediaStreamSource(stream);
        vadSourceRef.current = source;

        const analyser = ctx.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);
        vadAnalyserRef.current = analyser;

        const dataArray = new Uint8Array(analyser.frequencyBinCount);

        // VAD Loop
        window.clearInterval(vadIntervalRef.current);
        vadIntervalRef.current = window.setInterval(() => {
          // If already speaking, or TTS is active, or Rec is active, don't trigger
          if (
            ttsGuardRef.current.speaking ||
            ttsGuardRef.current.ignoreUntil > performance.now()
          )
            return;

          // Check if recognition is already active (approximate check via internal state or if we know we started it)
          // We'll trust the browser to throw error or handle partial overlap, but best to avoid calling start() if active.
          // Since we can't easily peek into SpeechRecognition state property, we rely on the `onend` callback to re-arm.

          analyser.getByteFrequencyData(dataArray);
          // Calculate average volume
          let sum = 0;
          for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
          const avg = sum / dataArray.length;

          // Threshold (adjustable). 20-30 is usually good for background noise vs speech.
          if (avg > 25) {
            const rec = speechRecRef.current;
            if (rec) {
              try {
                rec.start();
                // Pause VAD for a bit to let Rec take over?
                // Actually, Rec will fire 'onstart' which we can track.
                // But `start()` determines if it's active.
                // If we call start() while active, it throws.
              } catch (e: any) {
                // Ignore "already started" errors
              }
            }
          }
        }, 100);
      } catch (e) {
        console.warn("[VAD] Failed to init mic for VAD:", e);
      }
    })();

    return () => {
      window.clearInterval(vadIntervalRef.current);
      vadStreamRef.current?.getTracks().forEach((t) => t.stop());
      vadContextRef.current?.close();
    };
  }, [voiceOn]);

  // Inject a nicer font (Inter) if the project doesn't already provide it.
  useEffect(() => {
    const id = "neon-font-inter";
    if (document.getElementById(id)) return;
    const link = document.createElement("link");
    link.id = id;
    link.rel = "stylesheet";
    link.href =
      "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap";
    document.head.appendChild(link);
    return () => link.remove();
  }, []);

  useEffect(() => {
    const pick = () => {
      const all = window.speechSynthesis.getVoices();
      if (!all?.length) return;

      // 1) Priority: Exact match for Google US English
      let best = all.find((v) => v.name === "Google US English");

      // 2) Fallback: any non-Microsoft English voice
      if (!best) {
        const nonMsEn = all.filter(
          (v) =>
            !/^microsoft/i.test(v.name) &&
            v.lang?.toLowerCase().startsWith("en"),
        );
        best = nonMsEn[0];
      }

      // 3) Fallback: any English voice
      if (!best) {
        best = all.find((v) => v.lang?.toLowerCase().startsWith("en"));
      }

      // 4) Fallback: any non-Microsoft voice
      if (!best) {
        best = all.find((v) => !/^microsoft/i.test(v.name));
      }

      // 5) Final fallback: first available
      if (!best) best = all[0];

      voiceRef.current = best;
      console.log(
        "[TTS] Selected voice:",
        voiceRef.current?.name,
        voiceRef.current?.lang,
      );
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
    guard.ignoreUntil = performance.now() + 10000;
    window.speechSynthesis.cancel();
    setIsSpeaking(true);

    // [FIX] Prepend silence/period to prevent first-syllable cutoff
    const safeMsg = ". " + msg;
    const u = new SpeechSynthesisUtterance(safeMsg);
    u.rate = 1.0;
    u.pitch = 1.0;
    if (voiceRef.current) u.voice = voiceRef.current;

    u.onend = () => {
      guard.speaking = false;
      guard.suppressAutoRestart = false;
      setIsSpeaking(false);
      setTimeout(() => {
        guard.ignoreUntil = performance.now() + 100;
      }, 500);
    };

    u.onerror = () => {
      guard.speaking = false;
      setIsSpeaking(false);
    };

    window.speechSynthesis.speak(u);
  }, []);

  /* ------------------------------ canvas sizing ------------------------------ */
  const resizeCanvasesToVideo = useCallback(() => {
    const v = videoRef.current;
    const draw = drawRef.current;
    const ov = overlayRef.current;
    if (!v || !draw || !ov) return;

    const w = v.videoWidth;
    const h = v.videoHeight;
    if (!w || !h) return;

    // Mobile: force portrait aspect (canvas fills screen or 9:16 constraint)
    const mobile = isMobile();
    let targetW = w;
    let targetH = h;

    if (mobile) {
      // Force match window size for "Cover" effect
      targetW = window.innerWidth;
      targetH = window.innerHeight;
      setVideoAspect(targetW / targetH);
    } else {
      // Desktop: use actual video dimensions to avoid zoom
      // Get the rendered size of the video element
      const rect = v.getBoundingClientRect();
      targetW = Math.floor(rect.width);
      targetH = Math.floor(rect.height);
      setVideoAspect(w / h);
    }

    if (draw.width !== targetW || draw.height !== targetH) {
      draw.width = targetW;
      draw.height = targetH;
      needsFullRedrawRef.current = true;
    }
    if (ov.width !== targetW || ov.height !== targetH) {
      ov.width = targetW;
      ov.height = targetH;
    }
  }, []);

  /* ------------------------------ stroke rendering ------------------------------ */
  const paintStrokeSegment = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      color: string,
      a: StrokePoint,
      b: StrokePoint,
      glow: boolean,
    ) => {
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

      // reset only what matters
      ctx.shadowBlur = 0;
    },
    [],
  );

  const fullRedraw = useCallback(() => {
    const canvas = drawRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const glowOn = settingsRef.current.glow;
    for (const stroke of strokesRef.current) {
      if (stroke.points.length < 2) continue;
      for (let i = 1; i < stroke.points.length; i++) {
        const a = stroke.points[i - 1];
        const b = stroke.points[i];
        paintStrokeSegment(ctx, stroke.color, a, b, glowOn);
      }
    }
  }, [paintStrokeSegment]);

  const startStroke = useCallback(
    (color: string, p: Pt, t: number, thickness: number): string => {
      redoRef.current = [];
      const bb = emptyBBox();
      expandBBox(bb, p);
      const s: Stroke = {
        id: uid(),
        color,
        points: [{ x: p.x, y: p.y, t, w: thickness }],
        bbox: bb,
      };
      strokesRef.current.push(s); // ✅ no spread
      strokeByIdRef.current.set(s.id, s); // ✅ fast lookup
      return s.id;
    },
    [],
  );

  const addStrokePoint = useCallback(
    (strokeId: string, p: Pt, t: number, thickness: number) => {
      const s = strokeByIdRef.current.get(strokeId);
      if (!s) return;

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
    [paintStrokeSegment],
  );

  /* -------------------------- history & clipboard ------------------------- */
  const undo = useCallback(() => {
    const s = strokesRef.current;
    if (!s.length) return;
    const last = s.pop();
    if (last) {
      strokeByIdRef.current.delete(last.id);
      redoRef.current.push(last);
      needsFullRedrawRef.current = true;
      speak("Undo.");
    }
  }, [speak]);

  const redo = useCallback(() => {
    const r = redoRef.current;
    if (!r.length) return;
    const back = r.pop();
    if (back) {
      strokesRef.current.push(back);
      strokeByIdRef.current.set(back.id, back);
      needsFullRedrawRef.current = true;
      speak("Redo.");
    }
  }, [speak]);

  const clearAll = useCallback(() => {
    if (strokesRef.current.length === 0) return;
    strokesRef.current = [];
    redoRef.current = [];
    strokeByIdRef.current.clear();
    needsFullRedrawRef.current = true;
    speak("Canvas cleared.");
  }, [speak]);

  /* ------------------------------- Eraser Logic ------------------------------- */
  const eraseStrokeAt = useCallback((pt: Pt) => {
    const radius = 30; // Eraser radius
    const nextStrokes: Stroke[] = [];
    let changed = false;

    for (const s of strokesRef.current) {
      // Quick bbox check
      if (
        pt.x < s.bbox.minX - radius ||
        pt.x > s.bbox.maxX + radius ||
        pt.y < s.bbox.minY - radius ||
        pt.y > s.bbox.maxY + radius
      ) {
        nextStrokes.push(s);
        continue;
      }

      let currentSegment: StrokePoint[] = [];
      const segments: StrokePoint[][] = [];
      let strokeHit = false;

      for (const p of s.points) {
        if (dist(p, pt) > radius) {
          currentSegment.push(p);
        } else {
          strokeHit = true;
          if (currentSegment.length > 0) {
            segments.push(currentSegment);
            currentSegment = [];
          }
        }
      }
      if (currentSegment.length > 0) segments.push(currentSegment);

      if (strokeHit) {
        changed = true;
        for (const seg of segments) {
          if (seg.length > 1) {
            nextStrokes.push({
              ...s,
              id: uid(),
              points: seg,
              bbox: computeBBox(seg),
            });
          }
        }
      } else {
        nextStrokes.push(s);
      }
    }

    if (changed) {
      strokesRef.current = nextStrokes;
      strokeByIdRef.current = new Map(nextStrokes.map((s) => [s.id, s]));
      needsFullRedrawRef.current = true;
    }
  }, []);

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

  const translateStroke = useCallback(
    (strokeId: string, dx: number, dy: number) => {
      const s = strokeByIdRef.current.get(strokeId);
      if (!s) return;

      for (let i = 0; i < s.points.length; i++) {
        s.points[i].x += dx;
        s.points[i].y += dy;
      }
      s.bbox.minX += dx;
      s.bbox.maxX += dx;
      s.bbox.minY += dy;
      s.bbox.maxY += dy;

      needsFullRedrawRef.current = true;
    },
    [],
  );

  const rotateStroke = useCallback((strokeId: string, dAng: number) => {
    const s = strokeByIdRef.current.get(strokeId);
    if (!s) return;

    // centroid without allocations
    let sx = 0,
      sy = 0;
    for (let i = 0; i < s.points.length; i++) {
      sx += s.points[i].x;
      sy += s.points[i].y;
    }
    const c = { x: sx / s.points.length, y: sy / s.points.length };

    const bb = emptyBBox();
    const sin = Math.sin(dAng);
    const cos = Math.cos(dAng);

    for (let i = 0; i < s.points.length; i++) {
      const pt = s.points[i];
      const x = pt.x - c.x;
      const y = pt.y - c.y;
      pt.x = c.x + x * cos - y * sin;
      pt.y = c.y + x * sin + y * cos;
      expandBBox(bb, pt);
    }
    s.bbox = bb;

    needsFullRedrawRef.current = true;
  }, []);

  const scaleStroke = useCallback((strokeId: string, k: number) => {
    const s = strokeByIdRef.current.get(strokeId);
    if (!s) return;

    const kk = clamp(k, 0.25, 3.0);

    let sx = 0,
      sy = 0;
    for (let i = 0; i < s.points.length; i++) {
      sx += s.points[i].x;
      sy += s.points[i].y;
    }
    const c = { x: sx / s.points.length, y: sy / s.points.length };

    const bb = emptyBBox();
    for (let i = 0; i < s.points.length; i++) {
      const pt = s.points[i];
      pt.x = c.x + (pt.x - c.x) * kk;
      pt.y = c.y + (pt.y - c.y) * kk;
      pt.w = clamp(pt.w * kk, 1, 40);
      expandBBox(bb, pt);
    }
    s.bbox = bb;

    needsFullRedrawRef.current = true;
  }, []);

  /* ------------------------------ shapes ------------------------------ */
  const drawShapeAtCenter = useCallback(
    (
      shape: "circle" | "square" | "triangle" | "heart" | "star" | "arrow",
      size: "small" | "medium" | "big",
      color: string,
    ) => {
      const canvas = drawRef.current;
      if (!canvas) return;

      const minDim = Math.min(canvas.width, canvas.height);
      const r =
        size === "small"
          ? minDim * 0.11
          : size === "medium"
            ? minDim * 0.17
            : minDim * 0.24;
      const c: Pt = { x: canvas.width / 2, y: canvas.height / 2 };
      const t0 = performance.now();
      const thick = settingsRef.current.baseThickness;

      const pts: { x: number; y: number }[] = [];
      const add = (x: number, y: number) => pts.push({ x, y });

      if (shape === "circle") {
        for (let i = 0; i <= 72; i++) {
          const a = (i / 72) * Math.PI * 2;
          add(c.x + Math.cos(a) * r, c.y + Math.sin(a) * r);
        }
      } else if (shape === "square") {
        add(c.x - r, c.y - r);
        add(c.x + r, c.y - r);
        add(c.x + r, c.y + r);
        add(c.x - r, c.y + r);
        add(c.x - r, c.y - r);
      } else if (shape === "triangle") {
        add(c.x, c.y - r);
        add(c.x + r, c.y + r);
        add(c.x - r, c.y + r);
        add(c.x, c.y - r);
      } else if (shape === "heart") {
        for (let i = 0; i <= 72; i++) {
          const t = (i / 72) * Math.PI * 2;
          const x = 16 * Math.pow(Math.sin(t), 3);
          const y = -(
            13 * Math.cos(t) -
            5 * Math.cos(2 * t) -
            2 * Math.cos(3 * t) -
            Math.cos(4 * t)
          );
          // Normalize scale a bit
          add(c.x + (x / 16) * r, c.y + (y / 16) * r);
        }
      } else if (shape === "star") {
        const spikes = 5;
        const outer = r;
        const inner = r * 0.5;
        for (let i = 0; i <= spikes * 2; i++) {
          const rr = i % 2 === 0 ? outer : inner;
          const a = (i / (spikes * 2)) * Math.PI * 2 - Math.PI / 2;
          add(c.x + Math.cos(a) * rr, c.y + Math.sin(a) * rr);
        }
      } else if (shape === "arrow") {
        const w = r;
        const h = r * 0.6;
        add(c.x - w, c.y);
        add(c.x + w * 0.5, c.y);
        add(c.x + w * 0.5, c.y - h * 0.5);
        add(c.x + w, c.y);
        add(c.x + w * 0.5, c.y + h * 0.5);
        add(c.x + w * 0.5, c.y);
        add(c.x - w, c.y);
      }

      if (pts.length) {
        const s: Stroke = {
          id: uid(),
          points: pts.map((p) => ({ x: p.x, y: p.y, t: t0, w: thick })),
          color,
          bbox: computeBBox(
            pts.map((p) => ({ x: p.x, y: p.y, t: t0, w: thick })),
          ),
        };
        strokesRef.current.push(s);
        strokeByIdRef.current.set(s.id, s);
        needsFullRedrawRef.current = true;
        speak(`${size} ${shape} drawn`);
      }
    },
    [speak],
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

    return await new Promise((resolve) =>
      tmp.toBlob((b) => resolve(b), "image/png"),
    );
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

  const sharePng = useCallback(
    async (target?: "whatsapp" | "instagram" | "tiktok" | "facebook" | "x") => {
      const blob = await exportBlobMirrored();
      if (!blob) return;

      if (target) {
        await copyToClipboard();
        setTimeout(() => {
          let url = "";
          if (target === "whatsapp") url = "https://wa.me/";
          else if (target === "instagram") url = "https://www.instagram.com/";
          else if (target === "tiktok") url = "https://www.tiktok.com/";
          else if (target === "facebook") url = "https://www.facebook.com/";
          else if (target === "x") url = "https://twitter.com/intent/tweet";

          if (url) window.open(url, "_blank");
        }, 500);
        return;
      }

      const file = new File([blob], `neon_export_${Date.now()}.png`, {
        type: "image/png",
      });
      const nav = navigator as any;

      try {
        if (nav.canShare?.({ files: [file] }) && nav.share) {
          await nav.share({
            title: "Neon Studio",
            text: "Made with Neon Studio",
            files: [file],
          });
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
    },
    [copyToClipboard, downloadPng, exportBlobMirrored, speak],
  );

  /* ------------------------------ session export logic ------------------------------ */
  const handleSessionExport = useCallback(
    async (format: "gif" | "mp4" | "svg") => {
      speak(`Generating ${format.toUpperCase()}...`);

      // 1. Identify strokes to export (Session specific)
      const allStrokes = strokesRef.current;
      if (allStrokes.length === 0) {
        speak("Nothing to save.");
        return;
      }

      let startIndex = 0;
      if (lastSavedStrokeIdRef.current) {
        const idx = allStrokes.findIndex(
          (s) => s.id === lastSavedStrokeIdRef.current,
        );
        if (idx !== -1) startIndex = idx + 1;
      }

      const sessionStrokes = allStrokes.slice(startIndex);
      if (sessionStrokes.length === 0) {
        speak("No new drawings since last save.");
        return;
      }

      // 2. SVG Export (Vector)
      if (format === "svg") {
        const { width, height } = drawRef.current!;
        const svgHeader = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg" style="background:transparent">`;
        const svgFooter = "</svg>";
        let paths = "";

        for (const s of sessionStrokes) {
          if (s.points.length < 2) continue;
          const d = s.points
            .map(
              (p, i) =>
                `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`,
            )
            .join(" ");
          // Note: color and width are simplified here
          paths += `<path d="${d}" stroke="${s.color}" stroke-width="${s.points[0].w}" fill="none" stroke-linecap="round" stroke-linejoin="round" />`;
        }

        const blob = new Blob([svgHeader + paths + svgFooter], {
          type: "image/svg+xml",
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `neon_session_${Date.now()}.svg`;
        a.click();
        URL.revokeObjectURL(url);

        lastSavedStrokeIdRef.current =
          sessionStrokes[sessionStrokes.length - 1].id;
        speak("SVG Saved.");
        return;
      }

      // 3. GIF/MP4 via MediaRecorder (Replay)
      // Since we don't have gif.js or ffmpeg.wasm installed, we fallback to WebM/MP4 recording of a replay.
      // Ideally we would use a library for true GIF, but this meets the "Animation" requirement using available APIs.

      const offCanvas = document.createElement("canvas");
      offCanvas.width = drawRef.current!.width;
      offCanvas.height = drawRef.current!.height;
      const ctx = offCanvas.getContext("2d")!;

      // Paint background if needed (transparent for now)

      const stream = offCanvas.captureStream(30); // 30 FPS
      const mime = MediaRecorder.isTypeSupported("video/mp4")
        ? "video/mp4"
        : "video/webm";
      const recorder = new MediaRecorder(stream, {
        mimeType: mime,
        videoBitsPerSecond: 2500000,
      });

      const chunks: Blob[] = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: mime });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `neon_session_${Date.now()}.${mime === "video/mp4" ? "mp4" : "webm"}`;
        a.click();
        URL.revokeObjectURL(url);
        speak("Video saved.");
      };

      recorder.start();

      // REPLAY LOOP
      // We need to replay sessionStrokes over time.
      // To keep it fast, we'll speed up the replay or use a fixed time per stroke if timestamps are long.
      // Let's assume a "Review" speed (faster than real time).

      let tStart = performance.now();

      // Calculate total duration to normalize speed?
      // Simple approach: Render frame by frame.

      let strokeIdx = 0;
      let pointIdx = 1;

      const drawFrame = () => {
        // Draw a batch of points to simulate speed
        const BATCH = 3;

        for (let b = 0; b < BATCH; b++) {
          if (strokeIdx >= sessionStrokes.length) {
            recorder.stop();
            lastSavedStrokeIdRef.current =
              sessionStrokes[sessionStrokes.length - 1].id;
            return;
          }

          const s = sessionStrokes[strokeIdx];
          if (pointIdx >= s.points.length) {
            strokeIdx++;
            pointIdx = 1;
            continue;
          }

          // Draw s[pointIdx-1] to s[pointIdx]
          const p1 = s.points[pointIdx - 1];
          const p2 = s.points[pointIdx];

          ctx.lineCap = "round";
          ctx.lineJoin = "round";
          ctx.strokeStyle = s.color;
          ctx.lineWidth = p2.w;

          // Simple glow simulation
          if (settingsRef.current.glow) {
            ctx.shadowBlur = 10;
            ctx.shadowColor = s.color;
          } else {
            ctx.shadowBlur = 0;
          }

          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
          ctx.shadowBlur = 0;

          pointIdx++;
        }

        if (strokeIdx < sessionStrokes.length) {
          requestAnimationFrame(drawFrame);
        } else {
          recorder.stop();
          lastSavedStrokeIdRef.current =
            sessionStrokes[sessionStrokes.length - 1].id;
        }
      };

      requestAnimationFrame(drawFrame);
    },
    [speak],
  );

  /* ------------------------------ voice commands ------------------------------ */
  const effectiveColorForHand = useCallback((isAdmin: boolean) => {
    if (!isAdmin && Date.now() < guestUnlockUntilRef.current) return "#000000";
    return settingsRef.current.brushColor;
  }, []);

  /* -------------------------------- voice -------------------------------- */
  const textMatches = (text: string, phrases: string[]) =>
    phrases.some((p) => text.includes(p));

  const handleVoiceCommand = useCallback(
    (text: string) => {
      const t = text.toLowerCase();

      if (!t.includes("tensor")) {
        console.log("[Voice] Ignored - no 'Tensor' keyword:", t);
        return;
      }

      // STRICT EXPORT COMMANDS
      if (t.includes("export") || t.includes("share") || t.includes("send")) {
        if (textMatches(t, ["whatsapp"])) return void sharePng("whatsapp");
        if (textMatches(t, ["instagram", "insta"]))
          return void sharePng("instagram");
        if (textMatches(t, ["tiktok", "tic toc"]))
          return void sharePng("tiktok");
        if (textMatches(t, ["facebook", "fb"]))
          return void sharePng("facebook");
        if (textMatches(t, ["twitter", "x.com"])) return void sharePng("x");

        if (
          t.includes("png") ||
          t.includes("file") ||
          t.includes("download") ||
          t.includes("image")
        )
          return void downloadPng();
        if (t.includes("gif")) return void handleSessionExport("gif");
        if (t.includes("video") || t.includes("mp4") || t.includes("movie"))
          return void handleSessionExport("mp4");
        if (t.includes("svg") || t.includes("vector"))
          return void handleSessionExport("svg");

        return void sharePng(); // Default share
      }

      // CAMERA COMMANDS
      if (t.includes("camera") || t.includes("cam") || t.includes("view")) {
        if (
          t.includes("swap") ||
          t.includes("switch") ||
          t.includes("flip") ||
          t.includes("change")
        ) {
          setFacingMode((f: "user" | "environment") =>
            f === "user" ? "environment" : "user",
          );
          speak("Swapping camera.");
          return;
        }
      }

      // ERASER MODES
      // Type 1: Remove specific item or selection
      if (
        t.includes("remove") ||
        t.includes("delete") ||
        t.includes("erase") ||
        t.includes("clear")
      ) {
        if (
          t.includes("all") ||
          t.includes("canvas") ||
          t.includes("everything")
        ) {
          clearAll();
          return;
        }

        // AR Items
        if (t.includes("hat")) {
          setArItems((p) => ({ ...p, hat: false }));
          speak("Hat removed.");
          return;
        }
        if (t.includes("glasses")) {
          setArItems((p) => ({ ...p, glasses: false }));
          speak("Glasses removed.");
          return;
        }
      }

      // AR Add Commands
      if (t.includes("add") || t.includes("wear") || t.includes("put on")) {
        if (t.includes("hat")) {
          setArItems((p) => ({ ...p, hat: true }));
          speak("Hat added.");
          return;
        }
        if (t.includes("glasses") || t.includes("shades")) {
          setArItems((p) => ({ ...p, glasses: true }));
          speak("Glasses added.");
          return;
        }
      }

      // Type 2: Eraser Mode (Partial)
      if (t.includes("eraser mode") || t.includes("eraser tool")) {
        settingsRef.current.eraserMode = true;
        setEraserMode(true);
        speak("Eraser mode enabled. Use your index finger to erase.");
        return;
      }
      if (t.includes("draw mode") || t.includes("brush") || t.includes("pen")) {
        settingsRef.current.eraserMode = false;
        setEraserMode(false);
        speak("Draw mode enabled.");
        return;
      }

      // Shapes
      if (t.includes("draw") || t.includes("create") || t.includes("make")) {
        let shape = "";
        if (t.includes("circle")) shape = "circle";
        else if (t.includes("square") || t.includes("box")) shape = "square";
        else if (t.includes("triangle")) shape = "triangle";
        else if (t.includes("heart")) shape = "heart";
        else if (t.includes("star")) shape = "star";
        else if (t.includes("arrow")) shape = "arrow";

        if (shape) {
          const size =
            t.includes("big") || t.includes("large")
              ? "big"
              : t.includes("small") || t.includes("tiny")
                ? "small"
                : "medium";
          drawShapeAtCenter(shape as any, size, settingsRef.current.brushColor);
          return;
        }
      }

      // Toggles
      if (t.includes("glow")) {
        if (t.includes("on") || t.includes("enable")) {
          setGlow(true);
          speak("Glow on.");
        } else if (t.includes("off") || t.includes("disable")) {
          setGlow(false);
          speak("Glow off.");
        }
        return;
      }
      if (t.includes("skeleton") || t.includes("landmarks")) {
        if (t.includes("show") || t.includes("on")) {
          setShowLandmarks(true);
          speak("Skeleton visible.");
        } else {
          setShowLandmarks(false);
          speak("Skeleton hidden.");
        }
        return;
      }

      if (t.includes("undo")) return undo();
      if (t.includes("redo")) return redo();
      if (t.includes("copy")) return void copyToClipboard();

      // Brush properties
      if (t.includes("color") || t.includes("set brush")) {
        const m = t.match(/color (to )?(\S+)/i);
        if (m?.[2]) {
          setBrushColor(parseCssColorCandidate(m[2]));
          speak("Color set.");
          return;
        }
      }

      speak("Command not recognized.");
    },
    [
      clearAll,
      drawShapeAtCenter,
      downloadPng,
      redo,
      speak,
      undo,
      copyToClipboard,
      sharePng,
    ],
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
      const echo =
        guard.lastSpoken &&
        (ft === guard.lastSpoken || ft.includes(guard.lastSpoken));
      if (echo) return;

      if (ft && guard.lastCmd.text === ft && now - guard.lastCmd.at < 1800)
        return;
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
      // [FIX] Removed auto-restart. VAD will restart it when sound is detected.
    };

    rec.onerror = (e: any) => {
      console.warn("[Voice] Error:", e.error);
      if (e.error === "not-allowed")
        setVoiceHint("Mic blocked. Click Mic once to allow.");
      else if (e.error !== "no-speech") setVoiceHint(`Voice error: ${e.error}`);
    };
    return rec;
  }, []);

  /* ------------------------------ camera + model boot ------------------------------ */
  const isMobile = useCallback(() => {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
      navigator.userAgent,
    );
  }, []);

  const updateDevices = useCallback(async () => {
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      let video = all.filter((d) => d.kind === "videoinput");

      // On desktop, filter out phone/remote cameras
      if (!isMobile()) {
        const filtered = video.filter(
          (d) =>
            !/phone|mobile|remote|link to windows|redmi|xiaomi|samsung|huawei|oppo|vivo|oneplus|realme/i.test(
              d.label,
            ),
        );
        // Only use filtered list if we have at least one camera left
        if (filtered.length > 0) {
          video = filtered;
        }
      }

      setDevices(video);

      if (preferIntegrated && video.length > 0) {
        // Priority:
        // 1. OV/Azurewave integrated cameras (common in laptops)
        // 2. Built-in/Integrated/FaceTime labeled
        // 3. "User" facing
        // 4. First available
        const ov = video.find((d) =>
          /ov\d+|azurewave|integrated|built-in|facetime/i.test(d.label),
        );
        const userFacing = video.find((d) => /user|front/i.test(d.label));
        const target = ov || userFacing || video[0];

        if (target) setSelectedDeviceId(target.deviceId);
      }
    } catch (e) {
      console.warn("Device enumeration failed:", e);
    }
  }, [preferIntegrated]);

  useEffect(() => {
    updateDevices();
  }, [preferIntegrated, updateDevices]);

  const startCamera = useCallback(async () => {
    const v = videoRef.current;
    if (!v) return;

    try {
      console.log("[startCamera] Closing old stream if any...");
      const old = v.srcObject as MediaStream | null;
      old?.getTracks().forEach((tr) => tr.stop());
      v.srcObject = null;

      console.log("[startCamera] Requesting getUserMedia...");

      // [FIX] FOV Crop fix: Laptops use Landscape 16:9, Mobile uses Portrait 9:16
      const mobile = isMobile();
      const isPortraitWindow = window.innerHeight > window.innerWidth;

      // On desktop/laptop, we almost always want landscape to avoid sensor cropping (FOV loss)
      // unless the user specifies otherwise or it's a very specific mobile-like tablet.
      const usePortrait = mobile && isPortraitWindow;

      const idealW = usePortrait ? 720 : 1280;
      const idealH = usePortrait ? 1280 : 720;

      const constraints: MediaStreamConstraints = {
        video: {
          width: { ideal: idealW },
          height: { ideal: idealH },
        },
        audio: false,
      };

      if (selectedDeviceId) {
        (constraints.video as MediaTrackConstraints).deviceId = {
          exact: selectedDeviceId,
        };
      } else {
        (constraints.video as MediaTrackConstraints).facingMode = facingMode;
      }

      const stream = await navigator.mediaDevices.getUserMedia(constraints);

      v.muted = true;
      v.playsInline = true;
      v.srcObject = stream;

      console.log("[startCamera] Waiting for loadedmetadata or readyState...");

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
  }, [resizeCanvasesToVideo, facingMode, selectedDeviceId]);

  // Restart camera when device/facing selection changes
  useEffect(() => {
    if (ready) startCamera();
  }, [selectedDeviceId, facingMode, ready, startCamera]);

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

          pointFilter: new OneEuro2D(
            settingsRef.current.oneEuroMinCutoff,
            settingsRef.current.oneEuroBeta,
          ),
          pinchFilter: new OneEuro2D(
            settingsRef.current.oneEuroMinCutoff,
            settingsRef.current.oneEuroBeta,
          ),
          palmFilter: new OneEuro2D(
            settingsRef.current.oneEuroMinCutoff,
            settingsRef.current.oneEuroBeta,
          ),

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
    const survivors = prev.filter(
      (tr) => nowMs - tr.lastSeenMs < ttl && !used.has(tr.id),
    );
    tracksRef.current = [...next, ...survivors].slice(
      0,
      settingsRef.current.maxHands,
    );
  }, []);

  const updateAdminScaleFromTracks = useCallback(() => {
    const tracks = tracksRef.current;
    if (!tracks.length) return;
    const best = tracks.reduce((a, b) =>
      b.det.scalePx > a.det.scalePx ? b : a,
    );
    const target = best.det.scalePx;
    if (!adminScaleRef.current) adminScaleRef.current = target;
    adminScaleRef.current = adminScaleRef.current * 0.85 + target * 0.15;
  }, []);

  const isAdminGroup = useCallback((tr: Track) => {
    const s = adminScaleRef.current || tr.det.scalePx;
    const rel = Math.abs(tr.det.scalePx - s) / Math.max(1, s);
    return rel <= adminTolerance;
  }, []);

  const guestsUnlocked = useCallback(
    () => Date.now() < guestUnlockUntilRef.current,
    [],
  );
  const isAllowedToControl = useCallback(
    (tr: Track) => isAdminGroup(tr) || guestsUnlocked(),
    [guestsUnlocked, isAdminGroup],
  );

  const maybeUnlockGuestsByAdminBlackStrokeStart = useCallback(
    (tr: Track, color: string) => {
      if (isAdminGroup(tr) && color.toLowerCase() === "#000000") {
        guestUnlockUntilRef.current = Date.now() + 30_000;
        speak("Guests unlocked for thirty seconds. Guests draw in black.");
      }
    },
    [isAdminGroup, speak],
  );

  /* ------------------------------ interactions ------------------------------ */
  const processInteractions = useCallback(
    (nowMs: number) => {
      const canvas = drawRef.current;
      if (!canvas) return;

      const tracks = tracksRef.current;

      const act = tracks.map((tr) => {
        tr.pointFilter.set(
          settingsRef.current.oneEuroMinCutoff,
          settingsRef.current.oneEuroBeta,
        );
        tr.pinchFilter.set(
          settingsRef.current.oneEuroMinCutoff,
          settingsRef.current.oneEuroBeta,
        );
        tr.palmFilter.set(
          settingsRef.current.oneEuroMinCutoff,
          settingsRef.current.oneEuroBeta,
        );

        const det = tr.det;
        const allowed = isAllowedToControl(tr);

        tr.stablePoint = det.rawPoint
          ? Math.min(6, tr.stablePoint + 1)
          : Math.max(0, tr.stablePoint - 1);
        tr.stablePalm = det.rawPalm
          ? Math.min(6, tr.stablePalm + 1)
          : Math.max(0, tr.stablePalm - 1);

        const POINT_ACTIVE = tr.stablePoint >= 3;
        const PALM_ACTIVE = tr.stablePalm >= 3;

        const pinchOn = det.pinchStrength > 0.75 || det.rawGrab;
        const pinchOff = det.pinchStrength < 0.45 && !det.rawGrab;

        const pointPt = tr.pointFilter.filter(det.indexTip, nowMs);
        const pinchPt = tr.pinchFilter.filter(
          {
            x: (det.indexTip.x + det.thumbTip.x) / 2,
            y: (det.indexTip.y + det.thumbTip.y) / 2,
          },
          nowMs,
        );
        const palmPt = tr.palmFilter.filter(det.palm, nowMs);

        const pinchAng = Math.atan2(
          det.indexTip.y - det.thumbTip.y,
          det.indexTip.x - det.thumbTip.x,
        );

        return {
          tr,
          det,
          allowed,
          POINT_ACTIVE,
          PALM_ACTIVE,
          pinchOn,
          pinchOff,
          pointPt,
          pinchPt,
          palmPt,
          pinchAng,
        };
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
            a.tr.selectedStrokeId &&
            a.tr.selectedStrokeId === b.tr.selectedStrokeId
              ? a.tr.selectedStrokeId
              : "";

          tp.strokeId =
            sameSelected ||
            pickStroke(mid) ||
            strokesRef.current.at(-1)?.id ||
            "";
          tp.baseDist = d;
          tp.baseAng = ang;
          tp.lastK = 1;
          tp.lastAng = 0;
          tp.lastMid = mid;
        } else if (tp.strokeId) {
          const dx = mid.x - tp.lastMid.x;
          const dy = mid.y - tp.lastMid.y;
          tp.lastMid = mid;
          if (Math.abs(dx) + Math.abs(dy) > 0.2)
            translateStroke(tp.strokeId, dx, dy);

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
          const ang = Math.atan2(
            det.indexMcp.y - det.wrist.y,
            det.indexMcp.x - det.wrist.x,
          );
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
          if (Math.abs(dx) + Math.abs(dy) > 0.2)
            translateStroke(tr.selectedStrokeId, dx, dy);

          const dAng = normAngleDelta(x.pinchAng - tr.pinchStartAng);
          const relAng = normAngleDelta(dAng - tr.lastPinchAng);
          tr.lastPinchAng = dAng;
          if (Math.abs(relAng) > 0.012)
            rotateStroke(tr.selectedStrokeId, relAng);

          const rel = clamp(
            det.scalePx / Math.max(1, tr.pinchStartScalePx),
            0.92,
            1.08,
          );
          tr.pinchStartScalePx = det.scalePx;
          if (Math.abs(rel - 1) > 0.01) scaleStroke(tr.selectedStrokeId, rel);
        }

        if (!tr.isPointing && x.POINT_ACTIVE && !tr.isPinching) {
          tr.isPointing = true;

          if (settingsRef.current.eraserMode) {
            // Eraser mode just activates, doesn't start a stroke
            eraseStrokeAt(x.pointPt);
          } else {
            const isAdmin = isAdminGroup(tr);
            const color = effectiveColorForHand(isAdmin);
            maybeUnlockGuestsByAdminBlackStrokeStart(tr, color);

            const adminS = adminScaleRef.current || det.scalePx;
            const handScale = clamp(
              det.scalePx / Math.max(1, adminS),
              0.8,
              1.35,
            );
            const thickness = clamp(
              settingsRef.current.baseThickness * handScale,
              1,
              40,
            );

            tr.drawStrokeId = startStroke(color, x.pointPt, nowMs, thickness);
          }
        } else if (tr.isPointing && (!x.POINT_ACTIVE || tr.isPinching)) {
          tr.isPointing = false;
          tr.drawStrokeId = "";
        } else if (tr.isPointing) {
          if (settingsRef.current.eraserMode) {
            eraseStrokeAt(x.pointPt);
          } else if (tr.drawStrokeId) {
            const adminS = adminScaleRef.current || det.scalePx;
            const handScale = clamp(
              det.scalePx / Math.max(1, adminS),
              0.8,
              1.35,
            );
            const thickness = clamp(
              settingsRef.current.baseThickness * handScale +
                det.pinchStrength * 2.2,
              1,
              40,
            );
            addStrokePoint(tr.drawStrokeId, x.pointPt, nowMs, thickness);
          }
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
    ],
  );

  /* ------------------------------ overlay render ------------------------------ */
  const renderOverlay = useCallback(
    (faceRes?: FaceLandmarkerResult | null) => {
      const ov = overlayRef.current;
      const canvas = drawRef.current;
      if (!ov || !canvas) return;

      const ctx = ov.getContext("2d")!;
      ctx.clearRect(0, 0, ov.width, ov.height);

      const W = ov.width;
      const H = ov.height;

      const v = videoRef.current;
      const map = v
        ? getMapScale(
            v.videoWidth,
            v.videoHeight,
            W,
            H,
            isMobile() ? "cover" : "fill",
          )
        : { scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0 };

      // AR Rendering
      if (
        faceRes &&
        faceRes.faceLandmarks &&
        faceRes.faceLandmarks.length > 0
      ) {
        const face = faceRes.faceLandmarks[0];

        // Helper to get coords
        const get = (idx: number) => ({
          x: face[idx].x * v!.videoWidth * map.scaleX - map.offsetX,
          y: face[idx].y * v!.videoHeight * map.scaleY - map.offsetY,
        });

        // GLasses
        if (arItems.glasses) {
          const eyeL = get(33); // Left eye corner
          const eyeR = get(263); // Right eye corner
          const center = { x: (eyeL.x + eyeR.x) / 2, y: (eyeL.y + eyeR.y) / 2 };
          const size = dist(eyeL, eyeR) * 2.2;

          ctx.save();
          ctx.translate(center.x, center.y);
          // Simple rotation based on eyes
          const angle = Math.atan2(eyeR.y - eyeL.y, eyeR.x - eyeL.x);
          ctx.rotate(angle);

          // Draw Glasses (Neon Style)
          ctx.beginPath();
          ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
          ctx.strokeStyle = theme.accent;
          ctx.lineWidth = 4;

          // Lenses
          ctx.rect(-size / 2, -size / 5, size / 2.2, size / 2.5);
          ctx.rect(size / 22, -size / 5, size / 2.2, size / 2.5);
          ctx.fill();
          ctx.stroke();

          // Bridge
          ctx.beginPath();
          ctx.moveTo(-size / 25, -size / 5);
          ctx.lineTo(size / 25, -size / 5);
          ctx.stroke();

          ctx.restore();
        }

        // Hat
        if (arItems.hat) {
          const forehead = get(10); // Top of head
          const chin = get(152);
          const headHeight = dist(forehead, chin);
          const center = { x: forehead.x, y: forehead.y - headHeight * 0.3 };
          const size = headHeight * 1.5;

          ctx.save();
          ctx.translate(center.x, center.y);
          // Angle from chin to forehead
          const angle =
            Math.atan2(forehead.y - chin.y, forehead.x - chin.x) + Math.PI / 2;
          ctx.rotate(angle);

          // Draw Cyber Hat
          ctx.fillStyle = theme.accent;
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 3;

          ctx.beginPath();
          ctx.moveTo(-size / 2, 0); // Brim L
          ctx.lineTo(size / 2, 0); // Brim R
          ctx.lineTo(size / 3, -size / 2); // Top R
          ctx.lineTo(-size / 3, -size / 2); // Top L
          ctx.closePath();
          ctx.fill();
          ctx.stroke();

          // Neon Glow line
          ctx.beginPath();
          ctx.strokeStyle = "#0ff";
          ctx.moveTo(-size / 3, -size / 4);
          ctx.lineTo(size / 3, -size / 4);
          ctx.stroke();

          ctx.restore();
        }
      }

      // Drawing skeleton if enabled
      const tracks = tracksRef.current;
      if (settingsRef.current.showLandmarks) {
        for (const tr of tracks) {
          const lms = tr.det.rawLandmarks;
          if (!lms?.length) continue;

          const admin = isAdminGroup(tr);
          ctx.strokeStyle = admin ? "#69f0ae" : "#ffffff";
          ctx.lineWidth = 2;
          ctx.globalAlpha = 0.65;

          // connectors
          for (let i = 0; i < HAND_CONNECTIONS.length; i++) {
            const [a, b] = HAND_CONNECTIONS[i];
            const p1 = lms[a],
              p2 = lms[b];
            if (!p1 || !p2) continue;
            ctx.beginPath();
            ctx.moveTo(
              p1.x * v!.videoWidth * map.scaleX - map.offsetX,
              p1.y * v!.videoHeight * map.scaleY - map.offsetY,
            );
            ctx.lineTo(
              p2.x * v!.videoWidth * map.scaleX - map.offsetX,
              p2.y * v!.videoHeight * map.scaleY - map.offsetY,
            );
            ctx.stroke();
          }

          // landmarks (small dots)
          ctx.fillStyle = admin ? "#69f0ae" : "#ffffff";
          for (let i = 0; i < lms.length; i++) {
            const p = lms[i];
            ctx.beginPath();
            const cx = p.x * v!.videoWidth * map.scaleX - map.offsetX;
            const cy = p.y * v!.videoHeight * map.scaleY - map.offsetY;
            ctx.arc(cx, cy, 3, 0, Math.PI * 2);
            ctx.fill();
          }

          ctx.globalAlpha = 1;
        }
      }

      for (const tr of tracks) {
        const allowed = isAllowedToControl(tr);
        const admin = isAdminGroup(tr);
        const color = admin
          ? "rgba(105,240,174,0.45)"
          : allowed
            ? "rgba(255,204,102,0.4)"
            : "rgba(255,85,102,0.35)";

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
        const gesture =
          tr.det.gesture && tr.det.gesture !== "None"
            ? ` [${tr.det.gesture}]`
            : "";
        const conf = tr.det.gestureConfidence
          ? ` ${(tr.det.gestureConfidence * 100).toFixed(0)}%`
          : "";

        ctx.font = `bold 14px ${fontStack}`;
        ctx.fillStyle = admin
          ? "rgba(105,240,174,1.0)"
          : allowed
            ? "rgba(255,204,102,1.0)"
            : "rgba(255,85,102,1.0)";

        const tx = ov.width - tr.det.indexTip.x + 16;
        const ty = tr.det.indexTip.y - 12;
        ctx.fillText(label + gesture + conf, tx, ty);
        ctx.restore();
        ctx.restore();

        if (tr.selectedStrokeId) {
          const s = strokesRef.current.find(
            (x) => x.id === tr.selectedStrokeId,
          );
          if (s) {
            const bb = padBBox(s.bbox, 12);
            ctx.save();
            ctx.setLineDash([8, 6]);
            ctx.lineWidth = 2;
            ctx.strokeStyle = admin
              ? "rgba(105,240,174,0.9)"
              : "rgba(255,204,102,0.9)";
            ctx.strokeRect(
              bb.minX,
              bb.minY,
              bb.maxX - bb.minX,
              bb.maxY - bb.minY,
            );
            ctx.restore();
          }
        }
      }
    },
    [
      fontStack,
      guestsUnlocked,
      isAdminGroup,
      isAllowedToControl,
      arItems,
      theme,
    ],
  );

  useEffect(() => {
    let mounted = true;

    const onResize = () => {
      resizeCanvasesToVideo();
    };
    const onOrientationChange = () => {
      if (ready) startCamera();
    };

    window.addEventListener("resize", onResize);
    window.addEventListener("orientationchange", onOrientationChange);

    (async () => {
      setLoadingStep("1/4 Camera…");
      try {
        await updateDevices();
        await startCamera();
      } catch (e: any) {
        setLoadingStep(`Cam ERROR: ${e.name || "Access"}`);
        return;
      }

      setLoadingStep("2/4 Assets…");
      const loadWithTimeout = async <T,>(
        promise: Promise<T>,
        timeoutMs: number,
        label: string,
      ): Promise<T> => {
        return Promise.race([
          promise,
          new Promise<T>((_, reject) =>
            setTimeout(() => reject(new Error(`${label} timeout`)), timeoutMs),
          ),
        ]);
      };

      try {
        const vision = await loadWithTimeout(
          FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm",
          ),
          15000,
          "WASM",
        );

        const modelPath =
          "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";
        const faceModelPath =
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

        const tryCreate = async (delegate: "GPU" | "CPU") => {
          setLoadingStep(`3/4 Init ${delegate}…`);

          const gesturePromise = loadWithTimeout(
            GestureRecognizer.createFromOptions(vision, {
              baseOptions: { modelAssetPath: modelPath, delegate },
              runningMode: "VIDEO",
              numHands: settingsRef.current.maxHands,
              minHandDetectionConfidence: 0.5,
              minHandPresenceConfidence: 0.5,
              minTrackingConfidence: 0.5,
            }),
            15000,
            `Gesture-${delegate}`,
          );

          const facePromise = loadWithTimeout(
            FaceLandmarker.createFromOptions(vision, {
              baseOptions: { modelAssetPath: faceModelPath, delegate },
              runningMode: "VIDEO",
              numFaces: 1,
              minFaceDetectionConfidence: 0.5,
              minFacePresenceConfidence: 0.5,
              minTrackingConfidence: 0.5,
              outputFaceBlendshapes: true,
            }),
            15000,
            `Face-${delegate}`,
          );

          return Promise.all([gesturePromise, facePromise]);
        };

        let results;
        try {
          results = await tryCreate("GPU");
        } catch (e) {
          console.warn("GPU init failed, falling back to CPU...", e);
          results = await tryCreate("CPU");
        }

        if (!mounted) {
          results[0].close();
          results[1].close();
          return;
        }

        setLandmarker(results[0]);
        setFaceLandmarker(results[1]);
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
      window.removeEventListener("orientationchange", onOrientationChange);
      const stream = videoRef.current?.srcObject as MediaStream | null;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [resizeCanvasesToVideo, speak, startCamera, ready, updateDevices]);

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
      if (typeof vAny.requestVideoFrameCallback === "function")
        vAny.requestVideoFrameCallback(cb);
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

        ictx.drawImage(v, 0, 0, sz.w, sz.h);

        const t0 = performance.now();
        const ts = performance.now();

        // ✅ infer on the downscaled canvas
        const res: GestureRecognizerResult = lm.recognizeForVideo(infer, ts);

        // ✅ Face only when AR is enabled, and throttle
        const faceLm = faceLandmarker;
        if (faceLm && arEnabledRef.current) {
          // 10 fps face is enough for AR overlays
          const FACE_MIN_DT = 1000 / 10;
          if ((step as any)._lastFaceMs === undefined)
            (step as any)._lastFaceMs = 0;

          if (ts - (step as any)._lastFaceMs >= FACE_MIN_DT) {
            (step as any)._lastFaceMs = ts;
            faceResultsRef.current = faceLm.detectForVideo(infer, ts);
          }
        } else {
          faceResultsRef.current = null;
        }

        const t1 = performance.now();

        const inferMs = t1 - t0;
        emaInfer = emaInfer === 0 ? inferMs : emaInfer * 0.85 + inferMs * 0.15;

        const landmarks = res.landmarks || [];
        const handedness = (res.handedness ||
          (res as any).handednesses ||
          []) as any[];
        const gestureResults = res.gestures || [];

        const draw = drawRef.current;
        if (draw) {
          const W = draw.width;
          const H = draw.height;

          const dets: HandDet[] = [];

          const map = getMapScale(
            v.videoWidth,
            v.videoHeight,
            W,
            H,
            isMobile() ? "cover" : "fill",
          );

          // Helper for mapping
          const mx = (n: number) => n * v.videoWidth * map.scaleX - map.offsetX;
          const my = (n: number) =>
            n * v.videoHeight * map.scaleY - map.offsetY;

          for (let i = 0; i < landmarks.length; i++) {
            const hand = landmarks[i];
            if (!hand?.length) continue;

            const hd0 = handedness[i]?.[0];
            const handName = (hd0?.categoryName ?? "Unknown") as
              | "Left"
              | "Right"
              | "Unknown";

            const wrist = {
              x: mx(hand[IDX.wrist].x),
              y: my(hand[IDX.wrist].y),
            };
            const indexMcp = {
              x: mx(hand[IDX.indexMcp].x),
              y: my(hand[IDX.indexMcp].y),
            };
            const middleMcp = {
              x: mx(hand[IDX.middleMcp].x),
              y: my(hand[IDX.middleMcp].y),
            };

            const thumbTip = {
              x: mx(hand[IDX.thumbTip].x),
              y: my(hand[IDX.thumbTip].y),
            };
            const indexTip = {
              x: mx(hand[IDX.indexTip].x),
              y: my(hand[IDX.indexTip].y),
            };
            const indexPip = {
              x: mx(hand[IDX.indexPip].x),
              y: my(hand[IDX.indexPip].y),
            };

            const middleTip = {
              x: mx(hand[IDX.middleTip].x),
              y: my(hand[IDX.middleTip].y),
            };
            const middlePip = {
              x: mx(hand[IDX.middlePip].x),
              y: my(hand[IDX.middlePip].y),
            };

            const ringTip = {
              x: mx(hand[IDX.ringTip].x),
              y: my(hand[IDX.ringTip].y),
            };
            const ringPip = {
              x: mx(hand[IDX.ringPip].x),
              y: my(hand[IDX.ringPip].y),
            };

            const pinkyTip = {
              x: mx(hand[IDX.pinkyTip].x),
              y: my(hand[IDX.pinkyTip].y),
            };
            const pinkyPip = {
              x: mx(hand[IDX.pinkyPip].x),
              y: my(hand[IDX.pinkyPip].y),
            };

            const palm = {
              x: mx(
                (hand[IDX.wrist].x +
                  hand[IDX.indexMcp].x +
                  hand[IDX.middleMcp].x +
                  hand[IDX.ringMcp].x +
                  hand[IDX.pinkyMcp].x) /
                  5,
              ),
              y: my(
                (hand[IDX.wrist].y +
                  hand[IDX.indexMcp].y +
                  hand[IDX.middleMcp].y +
                  hand[IDX.ringMcp].y +
                  hand[IDX.pinkyMcp].y) /
                  5,
              ),
            };

            const scalePx = Math.max(60, dist(wrist, middleMcp));
            const pinchDist = dist(indexTip, thumbTip);
            const pinchRatio = pinchDist / Math.max(1, scalePx);
            const pinchStrength = clamp(
              (0.28 - pinchRatio) / (0.28 - 0.12),
              0,
              1,
            );

            const rawLandmarks = hand; // ✅ keep normalized landmarks, no allocation

            const gFull = gestureResults[i]?.[0];
            const gestureName = gFull?.categoryName ?? "None";
            const gestureScore = gFull?.score ?? 0;

            const handIndex = i; // renamed for clarity
            const base = {
              handIndex,
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
            } as Omit<
              HandDet,
              | "rawPoint"
              | "rawPalm"
              | "rawGrab"
              | "rawLandmarks"
              | "gesture"
              | "gestureConfidence"
            >;

            const g = computeGestureBooleans(base);

            // Override manual booleans with robust MediaPipe gestures
            const isPointing = gestureName === "Pointing_Up";
            const isPalm = gestureName === "Open_Palm";
            const isGrab =
              gestureName === "Closed_Fist" || gestureName === "Thumb_Up";

            dets.push({
              ...base,
              rawPoint: isPointing || g.rawPoint,
              rawPalm: isPalm || g.rawPalm,
              rawGrab: isGrab,
              rawLandmarks,
              gesture: gestureName,
              gestureConfidence: gestureScore,
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
  }, [
    assignTracks,
    guestsUnlocked,
    processInteractions,
    ready,
    landmarker,
    faceLandmarker,
    updateAdminScaleFromTracks,
  ]);

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
        fullRedraw();
      }
      renderOverlay(faceResultsRef.current);

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
  }, [fullRedraw, renderOverlay]);

  /* ------------------------------ voice toggle ------------------------------ */
  useEffect(() => {
    if (!voiceOn) {
      try {
        speechRecRef.current?.stop();
      } catch {}
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

  const onQuickShape = (
    shape: "circle" | "square" | "triangle" | "heart" | "star" | "arrow",
  ) => {
    drawShapeAtCenter(shape, "small", settingsRef.current.brushColor);
  };

  return (
    <div className="app-container">
      {/* ---------------- MOBILE LAYOUT ---------------- */}
      <div className="mobile-layout">
        {/* Top Bar */}
        <div className="top-bar">
          {ready ? (
            <span>Running. Point → draw. Fist → grab. Palm → select.</span>
          ) : (
            <span>{loadingStep}</span>
          )}
          <button
            className="btn icon-only"
            style={{ position: "absolute", right: 8, height: 28, width: 28 }}
            onClick={() => setShowMobileSettings((v) => !v)}
          >
            <Icons.Settings />
          </button>
        </div>

        {/* Mobile Settings (was in canvas-wrapper, now direct child of mobile-layout) */}

        {/* Mobile Settings Panel (Collapsible) */}
        {showMobileSettings && (
          <div
            className="mobile-settings card"
            style={{
              position: "absolute",
              top: 40,
              right: 10,
              left: 10,
              zIndex: 100,
              background: "rgba(10,10,15,0.95)",
            }}
          >
            <div className="label-row" style={{ fontWeight: 900 }}>
              System Config
              <button
                className="btn btn-danger"
                style={{ padding: "4px 8px" }}
                onClick={() => setShowMobileSettings(false)}
              >
                X
              </button>
            </div>
            <div className="label-row">
              <span>Accuracy</span>
              <select
                value={inferPreset}
                onChange={(e) =>
                  setInferPreset(e.target.value as InferencePreset)
                }
                style={{
                  background: "#222",
                  color: "#fff",
                  border: "1px solid #444",
                }}
              >
                <option value="FAST">Fast</option>
                <option value="BALANCED">Balanced</option>
                <option value="HIGH">High</option>
              </select>
            </div>
            <div className="label-row">
              <span>FPS Cap ({inferFpsCap})</span>
              <input
                type="range"
                min={10}
                max={30}
                step={1}
                value={inferFpsCap}
                onChange={(e) => setInferFpsCap(parseInt(e.target.value))}
              />
            </div>
            <div className="label-row">
              <span>Prefer Integrated</span>
              <button
                className={`btn ${preferIntegrated ? "btn-primary" : ""}`}
                style={{ padding: "2px 8px" }}
                onClick={() => setPreferIntegrated(!preferIntegrated)}
              >
                {preferIntegrated ? "ON" : "OFF"}
              </button>
            </div>
            <div className="label-row">
              <span>Cam Source</span>
              <select
                value={selectedDeviceId}
                onChange={(e) => {
                  setSelectedDeviceId(e.target.value);
                  setPreferIntegrated(false);
                }}
                style={{
                  background: "#222",
                  color: "#fff",
                  border: "1px solid #444",
                  maxWidth: 120,
                }}
              >
                <option value="">Default</option>
                {devices.map((d) => (
                  <option key={d.deviceId} value={d.deviceId}>
                    {d.label?.slice(0, 20) || "Cam"}
                  </option>
                ))}
              </select>
            </div>
            <div className="label-row">
              <span>Landmarks</span>
              <button
                className={`btn ${showLandmarks ? "btn-primary" : ""}`}
                onClick={() => setShowLandmarks(!showLandmarks)}
              >
                {showLandmarks ? "ON" : "OFF"}
              </button>
            </div>
          </div>
        )}

        {/* Start Indicator */}
        <div className="tensor-status-container">
          <div className={`tensor-indicator ${isSpeaking ? "speaking" : ""}`} />
        </div>

        {/* Left Stack */}
        <div className="stack-left">
          <button className="btn icon-only" onClick={undo}>
            <Icons.Undo />
          </button>
          <button className="btn icon-only" onClick={redo}>
            <Icons.Redo />
          </button>
          <button
            className={`btn icon-only ${eraserMode ? "btn-primary" : ""}`}
            onClick={() => {
              const nx = !eraserMode;
              settingsRef.current.eraserMode = nx;
              setEraserMode(nx);
              speak(nx ? "Eraser." : "Draw.");
            }}
          >
            <Icons.Eraser />
          </button>
          <button className="btn icon-only btn-danger" onClick={clearAll}>
            <Icons.Clear />
          </button>
          <button
            className="btn icon-only"
            onClick={() => void copyToClipboard()}
          >
            <Icons.Copy />
          </button>
        </div>

        {/* Right Stack */}
        <div className="stack-right">
          <button
            className="btn icon-only"
            style={{ color: "#25D366" }}
            onClick={() => void sharePng("whatsapp")}
          >
            <Icons.WhatsApp />
          </button>
          <button
            className="btn icon-only"
            style={{ color: "#E1306C" }}
            onClick={() => void sharePng("instagram")}
          >
            <Icons.Instagram />
          </button>
          <button
            className="btn icon-only"
            style={{ color: "#1877F2" }}
            onClick={() => void sharePng("facebook")}
          >
            <Icons.Facebook />
          </button>
          <button className="btn icon-only" onClick={() => void sharePng("x")}>
            <Icons.Twitter />
          </button>
          <button className="btn icon-only" onClick={() => void sharePng()}>
            <Icons.Share />
          </button>
        </div>

        {/* Bottom Corners */}
        <div className="corner-bl">
          <button
            className="btn icon-only"
            onClick={() => {
              setFacingMode((f) => (f === "user" ? "environment" : "user"));
            }}
          >
            <Icons.Cameraswitch />
          </button>
        </div>
        <div className="corner-br">
          <button
            className="btn btn-primary"
            style={{ borderRadius: 20, padding: "8px 16px" }}
            onClick={() => void downloadPng()}
          >
            <Icons.Save />
          </button>
        </div>
      </div>

      {/* ---------------- DESKTOP LAYOUT (Preserved but wrapped) ---------------- */}
      <div className="desktop-layout left-panel">
        {/* Original Desktop Cards... */}
        <div className="card">
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div
                style={{ fontSize: 20, fontWeight: 900, color: theme.accent }}
              >
                Tensor Studio
              </div>
            </div>
            <button
              className="btn btn-primary"
              style={{ padding: "6px 12px", fontSize: 11 }}
              onClick={() => setShowInstructions(true)}
            >
              INSTRUCTIONS
            </button>
          </div>
          <div style={{ fontSize: 11, color: theme.muted, marginTop: 10 }}>
            Protocol {ready ? "Active" : "Booting..."} • Voice Authorisation
            Required
          </div>
        </div>

        <div className="card">
          <div
            className="label-row"
            style={{ fontWeight: 900, color: theme.fg }}
          >
            Tools
          </div>
          <div className="tool-grid">
            <button className="btn" title="Undo" onClick={undo}>
              <Icons.Undo />
            </button>
            <button className="btn" title="Redo" onClick={redo}>
              <Icons.Redo />
            </button>
            <button
              className="btn btn-danger"
              title="Clear All"
              onClick={clearAll}
            >
              <Icons.Clear />
            </button>
          </div>
        </div>

        <div className="card">
          <button
            className={`btn ${eraserMode ? "btn-primary" : ""}`}
            style={{ width: "100%" }}
            onClick={() => {
              const nx = !eraserMode;
              settingsRef.current.eraserMode = nx;
              setEraserMode(nx);
              if (nx) speak("Eraser enabled.");
              else speak("Draw mode.");
            }}
          >
            {eraserMode ? "Eraser ON" : "Eraser Mode"}
          </button>
        </div>

        <div className="card">
          <div className="tool-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
            <button
              className="btn btn-primary"
              onClick={() => void downloadPng()}
              style={{ gap: 6 }}
            >
              <Icons.Export /> PNG
            </button>
            <button
              className="btn btn-primary"
              onClick={() => void sharePng()}
              style={{ gap: 6 }}
            >
              <Icons.Share /> Share
            </button>
          </div>
        </div>

        <div className="card">
          <div
            className="label-row"
            style={{ fontWeight: 900, color: theme.fg }}
          >
            Appearance
          </div>
          <div className="label-row">
            <span>Brush Color</span>
            <input
              type="color"
              value={brushColor.startsWith("#") ? brushColor : "#69f0ae"}
              onChange={(e) => setBrushColor(e.target.value)}
              style={{
                width: 44,
                height: 28,
                background: "transparent",
                border: "none",
                cursor: "pointer",
              }}
            />
          </div>
          <div className="label-row">
            <span>Thickness</span>
            <input
              type="range"
              min={1}
              max={40}
              step={0.5}
              value={baseThickness}
              onChange={(e) => setBaseThickness(parseFloat(e.target.value))}
              style={{ flex: 1 }}
            />
            <span
              style={{ width: 40, textAlign: "right", color: theme.accent }}
            >
              {baseThickness.toFixed(0)}
            </span>
          </div>
          <div style={{ height: 16 }} />
          <div
            className="label-row"
            style={{ fontWeight: 900, color: theme.fg }}
          >
            Quick Shapes
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr 1fr",
              gap: 6,
            }}
          >
            <button
              className="btn icon-btn"
              onClick={() => onQuickShape("circle")}
            >
              <Icons.Circle />
            </button>
            <button
              className="btn icon-btn"
              onClick={() => onQuickShape("square")}
            >
              <Icons.Square />
            </button>
            <button
              className="btn icon-btn"
              onClick={() => onQuickShape("star")}
            >
              <Icons.Star />
            </button>
            <button
              className="btn icon-btn"
              onClick={() => onQuickShape("heart")}
            >
              <Icons.Heart />
            </button>
            <button
              className="btn icon-btn"
              onClick={() => onQuickShape("arrow")}
            >
              <Icons.Arrow />
            </button>
            <button
              className="btn icon-btn"
              onClick={() => onQuickShape("triangle")}
            >
              <Icons.Triangle />
            </button>
          </div>
          <div style={{ height: 12 }} />
          <div style={{ display: "flex", gap: 8 }}>
            <button
              className="btn"
              style={{ flex: 1, gap: 6 }}
              onClick={() => setGlow((v) => !v)}
            >
              <Icons.Glow /> {glow ? "On" : "Off"}
            </button>
            <button
              className="btn"
              style={{ flex: 1, gap: 6 }}
              onClick={() => setShowLandmarks((v) => !v)}
            >
              <Icons.Landmarks /> {showLandmarks ? "On" : "Off"}
            </button>
          </div>
        </div>

        <div className="card">
          <div
            className="label-row"
            style={{ fontWeight: 900, color: theme.fg }}
          >
            System Config
          </div>
          <div className="label-row">
            <span>Accuracy</span>
            <select
              value={inferPreset}
              onChange={(e) =>
                setInferPreset(e.target.value as InferencePreset)
              }
              style={{
                background: "rgba(255,255,255,0.05)",
                border: "1px solid var(--border)",
                borderRadius: 8,
                color: theme.fg,
                padding: "4px 8px",
              }}
            >
              <option value="FAST">Fast</option>
              <option value="BALANCED">Balanced</option>
              <option value="HIGH">High</option>
            </select>
          </div>
          <div className="label-row">
            <span>FPS Cap</span>
            <input
              type="range"
              min={10}
              max={30}
              step={1}
              value={inferFpsCap}
              onChange={(e) => setInferFpsCap(parseInt(e.target.value, 10))}
            />
            <span style={{ width: 24, textAlign: "right" }}>{inferFpsCap}</span>
          </div>
          <div style={{ height: 12 }} />
          <div
            className="label-row"
            style={{ fontWeight: 900, color: theme.fg }}
          >
            Camera Selection
          </div>
          <div className="label-row">
            <span>Prefer Integrated</span>
            <button
              className={`btn ${preferIntegrated ? "btn-primary" : ""}`}
              style={{ padding: "4px 8px", fontSize: 10 }}
              onClick={() => setPreferIntegrated(!preferIntegrated)}
            >
              {preferIntegrated ? "ON" : "OFF"}
            </button>
          </div>
          <div className="label-row">
            <span>Source</span>
            <select
              value={selectedDeviceId}
              onChange={(e) => {
                setSelectedDeviceId(e.target.value);
                if (preferIntegrated) setPreferIntegrated(false);
              }}
              style={{
                background: "rgba(255,255,255,0.05)",
                border: "1px solid var(--border)",
                borderRadius: 8,
                color: theme.fg,
                padding: "4px 8px",
                maxWidth: "140px",
                fontSize: 10,
              }}
            >
              <option value="">Default</option>
              {devices.map((d) => (
                <option key={d.deviceId} value={d.deviceId}>
                  {d.label || `Camera ${d.deviceId.slice(0, 4)}`}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Desktop Canvas is redundant here as we moved it to Mobile Layout (shared) or we need a Desktop Render? 
          Actually, the prompt implies "Current Mistakes... correct Implementation Plan". 
          The Plan said "Refactor JSX to use explicit mobile-layout and desktop-layout".
          The Canvas Wrapper is shared. But on Mobile it's absolute. On desktop it's in the grid.
      */}
      <div
        className={`canvas-wrapper ${tracksRef.current.length > 0 ? "visible" : ""}`}
      >
        <video ref={videoRef} className="video-layer" muted playsInline />
        <canvas ref={drawRef} className="canvas-layer" />
        <canvas ref={overlayRef} className="canvas-layer" />
        <canvas ref={inferCanvasRef} style={{ display: "none" }} />
        {!ready && (
          <div
            className="opening-overlay"
            style={{ background: "transparent" }}
          >
            <div className="nova-logo" />
          </div>
        )}
      </div>

      {/* Shared Canvas Wrapper mounted once */}
      {/* But wait, my CSS defines .mobile-layout as Flex and .desktop-layout as Grid columns 
        The css for .app-container on desktop is "grid 340px 1fr".
        So the canvas wrapper *must* be the second child of app-container for desktop grid work.
    */}
      {/* Fix:
       The Canvas Wrapper should be a direct child of app-container.
       Mobile CSS hides .desktop-layout (sidebar).
       Mobile CSS styles .canvas-wrapper as absolute centered.
       Desktop CSS styles .canvas-wrapper as grid item.
    */}
    </div>
  );
}
