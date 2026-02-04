import { useEffect, useMemo, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

/**
 * Landmark indices (MediaPipe Hands):
 * 0 wrist
 * 4 thumb tip, 8 index tip
 * 6 index PIP, 10 middle PIP, 14 ring PIP, 18 pinky PIP
 * 8,12,16,20 tips
 */
const IDX = {
  wrist: 0,
  thumbTip: 4,
  indexTip: 8,
  indexPip: 6,
  middleTip: 12,
  middlePip: 10,
  ringTip: 16,
  ringPip: 14,
  pinkyTip: 20,
  pinkyPip: 18,
  indexMcp: 5,
  middleMcp: 9,
  ringMcp: 13,
  pinkyMcp: 17,
} as const;

type Pt = { x: number; y: number };
type StrokePoint = { x: number; y: number; t: number; w: number };
type Stroke = { id: string; color: string; points: StrokePoint[] };

function clamp(v: number, a: number, b: number) {
  return Math.max(a, Math.min(b, v));
}
function dist(a: Pt, b: Pt) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
function uid() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16);
}
function rotatePoint(p: Pt, c: Pt, ang: number): Pt {
  const s = Math.sin(ang);
  const co = Math.cos(ang);
  const x = p.x - c.x;
  const y = p.y - c.y;
  return { x: c.x + x * co - y * s, y: c.y + x * s + y * co };
}
function centroid(points: Pt[]): Pt {
  if (points.length === 0) return { x: 0, y: 0 };
  let sx = 0,
    sy = 0;
  for (const p of points) {
    sx += p.x;
    sy += p.y;
  }
  return { x: sx / points.length, y: sy / points.length };
}

type CameraPreset = "FAST" | "BALANCED" | "HIGH";

function getPresetConstraints(preset: CameraPreset) {
  // Lower res = faster inference.
  // Using exact constraints isn’t guaranteed, but it helps most webcams.
  if (preset === "FAST") return { width: { ideal: 640 }, height: { ideal: 360 }, frameRate: { ideal: 30, max: 30 } };
  if (preset === "BALANCED") return { width: { ideal: 960 }, height: { ideal: 540 }, frameRate: { ideal: 30, max: 30 } };
  return { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 30 } };
}

export default function App() {
  const theme = useMemo(
    () => ({
      bg: "#0B0D10",
      panel: "rgba(0,0,0,0.35)",
      border: "rgba(255,255,255,0.12)",
      border2: "rgba(255,255,255,0.18)",
      fg: "#fff",
      muted: "rgba(255,255,255,0.70)",
      accent: "#69F0AE",
      danger: "#ff5566",
      warn: "#ffcc66",
    }),
    []
  );

  // DOM
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const drawRef = useRef<HTMLCanvasElement | null>(null);

  // Mediapipe
  const landmarkerRef = useRef<HandLandmarker | null>(null);

  // Drawing data
  const strokesRef = useRef<Stroke[]>([]);
  const redoRef = useRef<Stroke[]>([]);
  const selectionRef = useRef<string>("");

  // App state
  const [ready, setReady] = useState(false);
  const [fps, setFps] = useState(0);
  const [inferFpsCap, setInferFpsCap] = useState(24); // speed knob
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>("FAST");

  // UX
  const [showLandmarks, setShowLandmarks] = useState(false);
  const [glow, setGlow] = useState(true);
  const [smoothing, setSmoothing] = useState(0.7);
  const [baseThickness, setBaseThickness] = useState(6);
  const [brushColor, setBrushColor] = useState("#69F0AE");

  // Voice
  const [voiceOn, setVoiceOn] = useState(false);
  const [voiceHint, setVoiceHint] = useState<string>("Say: “change color to red”, “undo”, “clear”, “export”");
  const recogRef = useRef<any>(null);

  // Gesture state
  const gRef = useRef({
    pinch: false,
    openGrab: false,
    pointer: { x: 0, y: 0 } as Pt,
    smoothed: { x: 0, y: 0 } as Pt,
    pinchStrength: 0,
    lastInferMs: 0,
    lastVideoTime: -1,
    // manipulation
    mode: "none" as "none" | "draw" | "drag" | "rotate" | "grab",
    dragStrokeId: "",
    dragOffset: { x: 0, y: 0 } as Pt,
    lastAngle: 0,
    lastPointer: { x: 0, y: 0 } as Pt,
  });

  const [strokeCount, setStrokeCount] = useState(0);

  function resizeCanvases() {
    const video = videoRef.current;
    const overlay = overlayRef.current;
    const draw = drawRef.current;
    if (!video || !overlay || !draw) return;

    const w = video.videoWidth || 1280;
    const h = video.videoHeight || 720;

    overlay.width = w;
    overlay.height = h;
    draw.width = w;
    draw.height = h;

    redrawAll();
  }

  function redrawAll() {
    const draw = drawRef.current;
    if (!draw) return;
    const ctx = draw.getContext("2d")!;
    ctx.clearRect(0, 0, draw.width, draw.height);
    for (const s of strokesRef.current) paintStroke(ctx, s);
  }

  function paintStroke(ctx: CanvasRenderingContext2D, s: Stroke) {
    if (s.points.length < 2) return;
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = s.color;

    if (glow) {
      ctx.shadowBlur = 18;
      ctx.shadowColor = s.color;
    } else {
      ctx.shadowBlur = 0;
    }

    for (let i = 1; i < s.points.length; i++) {
      const a = s.points[i - 1];
      const b = s.points[i];
      ctx.lineWidth = b.w;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
    ctx.restore();
  }

  function startStroke(p: Pt, t: number, thickness: number) {
    redoRef.current = []; // invalidate redo
    const s: Stroke = { id: uid(), color: brushColor, points: [{ x: p.x, y: p.y, t, w: thickness }] };
    strokesRef.current = [...strokesRef.current, s];
    selectionRef.current = s.id;
    setStrokeCount(strokesRef.current.length);
  }

  function addStrokePoint(p: Pt, t: number, thickness: number) {
    const strokes = strokesRef.current;
    if (strokes.length === 0) return;
    const s = strokes[strokes.length - 1];
    s.points.push({ x: p.x, y: p.y, t, w: thickness });

    // incremental paint (fast)
    const ctx = drawRef.current?.getContext("2d");
    if (ctx && s.points.length >= 2) {
      const seg: Stroke = { ...s, points: s.points.slice(-2) };
      paintStroke(ctx, seg);
    }
  }

  function clearAll() {
    strokesRef.current = [];
    redoRef.current = [];
    selectionRef.current = "";
    setStrokeCount(0);
    const draw = drawRef.current;
    if (draw) draw.getContext("2d")!.clearRect(0, 0, draw.width, draw.height);
  }

  function undo() {
    const st = strokesRef.current;
    if (st.length === 0) return;
    const last = st[st.length - 1];
    redoRef.current = [last, ...redoRef.current];
    strokesRef.current = st.slice(0, -1);
    selectionRef.current = strokesRef.current.at(-1)?.id ?? "";
    setStrokeCount(strokesRef.current.length);
    redrawAll();
  }

  function redo() {
    const r = redoRef.current;
    if (r.length === 0) return;
    const [first, ...rest] = r;
    redoRef.current = rest;
    strokesRef.current = [...strokesRef.current, first];
    selectionRef.current = first.id;
    setStrokeCount(strokesRef.current.length);
    redrawAll();
  }

  function exportPng() {
    const draw = drawRef.current;
    if (!draw) return;
    draw.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `neon_export_${Date.now()}.png`;
      a.click();
      URL.revokeObjectURL(url);
    }, "image/png");
  }

  function pickStroke(p: Pt): { id: string; offset: Pt } | null {
    const threshold = 22;
    let best: { id: string; d: number; offset: Pt } | null = null;

    for (const s of strokesRef.current) {
      for (const pt of s.points) {
        const d = Math.hypot(pt.x - p.x, pt.y - p.y);
        if (d < threshold && (!best || d < best.d)) {
          best = { id: s.id, d, offset: { x: pt.x - p.x, y: pt.y - p.y } };
        }
      }
    }
    return best ? { id: best.id, offset: best.offset } : null;
  }

  function translateStroke(strokeId: string, dx: number, dy: number) {
    strokesRef.current = strokesRef.current.map((s) => {
      if (s.id !== strokeId) return s;
      return { ...s, points: s.points.map((pt) => ({ ...pt, x: pt.x + dx, y: pt.y + dy })) };
    });
    redrawAll();
  }

  function rotateStroke(strokeId: string, angle: number) {
    strokesRef.current = strokesRef.current.map((s) => {
      if (s.id !== strokeId) return s;
      const c = centroid(s.points.map((p) => ({ x: p.x, y: p.y })));
      return {
        ...s,
        points: s.points.map((pt) => {
          const rp = rotatePoint({ x: pt.x, y: pt.y }, c, angle);
          return { ...pt, x: rp.x, y: rp.y };
        }),
      };
    });
    redrawAll();
  }

  function drawSelectionOverlay(ctx: CanvasRenderingContext2D, strokeId: string) {
    const s = strokesRef.current.find((x) => x.id === strokeId);
    if (!s || s.points.length === 0) return;
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;
    for (const p of s.points) {
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x);
      maxY = Math.max(maxY, p.y);
    }
    ctx.save();
    ctx.strokeStyle = "rgba(105,240,174,0.85)";
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 6]);
    ctx.strokeRect(minX - 12, minY - 12, (maxX - minX) + 24, (maxY - minY) + 24);
    ctx.restore();
  }

  // --- Voice (Web Speech API) ---
  // SpeechRecognition is a standard browser API (Chrome is the most reliable).
  function setupVoice() {
    const w = window as any;
    const SR = w.SpeechRecognition || w.webkitSpeechRecognition;
    if (!SR) {
      setVoiceHint("Voice not supported in this browser. Try Chrome/Edge.");
      return null;
    }
    const rec = new SR();
    rec.lang = "en-US";
    rec.continuous = true;
    rec.interimResults = true;

    rec.onresult = (event: any) => {
      let finalText = "";
      let interim = "";

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const res = event.results[i];
        const txt = res[0].transcript.trim();
        if (res.isFinal) finalText += (finalText ? " " : "") + txt;
        else interim += (interim ? " " : "") + txt;
      }

      if (interim) setVoiceHint(`Heard: "${interim}"`);
      if (finalText) {
        setVoiceHint(`Command: "${finalText}"`);
        handleVoiceCommand(finalText);
      }
    };

    // Many browsers end recognition automatically; restarting on end keeps it “continuous” in practice.
    rec.onend = () => {
      if (voiceOn) {
        try {
          rec.start();
        } catch { }
      }
    };

    rec.onerror = (e: any) => {
      setVoiceHint(`Voice error: ${e?.error ?? "unknown"}`);
    };

    return rec;
  }

  function handleVoiceCommand(text: string) {
    const t = text.toLowerCase();

    const colorMatch = t.match(/(change|set)\s+(the\s+)?color\s+(to)\s+([#a-z0-9]+)/i);
    if (colorMatch?.[4]) {
      const c = colorMatch[4].trim();
      // allow css color words or hex
      setBrushColor(c.startsWith("#") ? c : c);
      return;
    }

    const thickMatch = t.match(/(thickness|brush\s*size)\s+(to\s+)?(\d+)/i);
    if (thickMatch?.[3]) {
      setBaseThickness(clamp(parseInt(thickMatch[3], 10), 2, 24));
      return;
    }

    if (t.includes("undo")) return undo();
    if (t.includes("redo")) return redo();
    if (t.includes("clear")) return clearAll();
    if (t.includes("export") || t.includes("save")) return exportPng();

    if (t.includes("show landmarks")) return setShowLandmarks(true);
    if (t.includes("hide landmarks")) return setShowLandmarks(false);

    if (t.includes("glow on")) return setGlow(true);
    if (t.includes("glow off")) return setGlow(false);

    if (t.includes("quality fast")) return setCameraPreset("FAST");
    if (t.includes("quality balanced")) return setCameraPreset("BALANCED");
    if (t.includes("quality high")) return setCameraPreset("HIGH");

    if (t.includes("faster")) return setInferFpsCap(15);
    if (t.includes("smoother")) return setSmoothing(0.8);

    setVoiceHint("No recognized command. Try: “change color to red”, “undo”, “clear”, “export”.");
  }

  // --- Camera boot / restart ---
  async function startCamera(preset: CameraPreset) {
    const video = videoRef.current;
    if (!video) return;

    // stop old tracks
    const old = video.srcObject as MediaStream | null;
    old?.getTracks().forEach((tr) => tr.stop());

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", ...getPresetConstraints(preset) },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();
    resizeCanvases();
  }

  // --- Main loop ---
  useEffect(() => {
    let mounted = true;

    const boot = async () => {
      await startCamera(cameraPreset);

      const vision = await FilesetResolver.forVisionTasks(
        // You can host wasm locally for faster load.
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm"
      );

      landmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      setReady(true);

      // Resize on layout changes
      const onResize = () => resizeCanvases();
      window.addEventListener("resize", onResize);

      // FPS EMA
      let last = performance.now();
      let ema = 0;

      const loop = () => {
        if (!mounted) return;

        const now = performance.now();
        const dt = Math.max(1, now - last);
        const inst = 1000 / dt;
        ema = ema === 0 ? inst : ema * 0.9 + inst * 0.1;
        last = now;
        setFps(ema);

        const video = videoRef.current;
        const overlay = overlayRef.current;
        const landmarker = landmarkerRef.current;

        if (video && overlay && landmarker) {
          const g = gRef.current;
          const overlayCtx = overlay.getContext("2d")!;
          overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

          // Performance optimization:
          // MediaPipe docs note detectForVideo is synchronous and blocks UI thread,
          // and recommend techniques like workers for heavy use.
          // We also (a) cap inference fps and (b) skip duplicate frames.
          const minInferDt = 1000 / Math.max(5, inferFpsCap);
          const videoTime = video.currentTime;

          let res: any = null;
          if (videoTime !== g.lastVideoTime && (now - g.lastInferMs) >= minInferDt) {
            g.lastVideoTime = videoTime;
            g.lastInferMs = now;
            // detectForVideo signature expects (videoFrame, timestamp)
            res = landmarker.detectForVideo(video, now);
          }

          // If we didn’t infer this frame, keep interaction stable by reusing last gesture state only.
          if (res?.landmarks?.[0]) {
            const hand = res.landmarks[0];

            // optional landmarks
            if (showLandmarks) {
              overlayCtx.fillStyle = "rgba(105,240,174,0.85)";
              for (const p of hand) {
                overlayCtx.beginPath();
                overlayCtx.arc(p.x * overlay.width, p.y * overlay.height, 4, 0, Math.PI * 2);
                overlayCtx.fill();
              }
            }

            // build key points
            const wrist = { x: hand[IDX.wrist].x * overlay.width, y: hand[IDX.wrist].y * overlay.height };
            const indexMcp = { x: hand[IDX.indexMcp].x * overlay.width, y: hand[IDX.indexMcp].y * overlay.height };
            const thumb = { x: hand[IDX.thumbTip].x * overlay.width, y: hand[IDX.thumbTip].y * overlay.height };
            const index = { x: hand[IDX.indexTip].x * overlay.width, y: hand[IDX.indexTip].y * overlay.height };

            const scale = Math.max(60, dist(wrist, indexMcp));
            const pinchDist = dist(thumb, index);
            const pinchStrength = clamp(1 - pinchDist / (scale * 0.55), 0, 1);

            const pointer = { x: (thumb.x + index.x) / 2, y: (thumb.y + index.y) / 2 };
            g.pinchStrength = pinchStrength;
            g.pointer = pointer;
            g.smoothed = {
              x: g.smoothed.x * smoothing + pointer.x * (1 - smoothing),
              y: g.smoothed.y * smoothing + pointer.y * (1 - smoothing),
            };

            // Open palm heuristic:
            // A finger is "extended" if its tip is farther from wrist than its PIP.
            const indexPip = { x: hand[IDX.indexPip].x * overlay.width, y: hand[IDX.indexPip].y * overlay.height };
            const middleTip = { x: hand[IDX.middleTip].x * overlay.width, y: hand[IDX.middleTip].y * overlay.height };
            const middlePip = { x: hand[IDX.middlePip].x * overlay.width, y: hand[IDX.middlePip].y * overlay.height };
            const ringTip = { x: hand[IDX.ringTip].x * overlay.width, y: hand[IDX.ringTip].y * overlay.height };
            const ringPip = { x: hand[IDX.ringPip].x * overlay.width, y: hand[IDX.ringPip].y * overlay.height };
            const pinkyTip = { x: hand[IDX.pinkyTip].x * overlay.width, y: hand[IDX.pinkyTip].y * overlay.height };
            const pinkyPip = { x: hand[IDX.pinkyPip].x * overlay.width, y: hand[IDX.pinkyPip].y * overlay.height };

            const extIndex = dist(index, wrist) > dist(indexPip, wrist) + 10;
            const extMiddle = dist(middleTip, wrist) > dist(middlePip, wrist) + 10;
            const extRing = dist(ringTip, wrist) > dist(ringPip, wrist) + 10;
            const extPinky = dist(pinkyTip, wrist) > dist(pinkyPip, wrist) + 10;
            const openPalm = extIndex && extMiddle && extRing && extPinky && pinchStrength < 0.25;

            // Palm center for grab
            const palm = {
              x:
                (hand[IDX.wrist].x + hand[IDX.indexMcp].x + hand[IDX.middleMcp].x + hand[IDX.ringMcp].x + hand[IDX.pinkyMcp].x) /
                5 *
                overlay.width,
              y:
                (hand[IDX.wrist].y + hand[IDX.indexMcp].y + hand[IDX.middleMcp].y + hand[IDX.ringMcp].y + hand[IDX.pinkyMcp].y) /
                5 *
                overlay.height,
            };

            // Cursor
            overlayCtx.save();
            overlayCtx.globalAlpha = 0.9;
            overlayCtx.fillStyle = openPalm ? "rgba(255,204,102,0.25)" : "rgba(105,240,174,0.25)";
            const r = openPalm ? 18 : 10 + 10 * pinchStrength;
            overlayCtx.beginPath();
            overlayCtx.arc(g.smoothed.x, g.smoothed.y, r, 0, Math.PI * 2);
            overlayCtx.fill();
            overlayCtx.restore();

            // Selection overlay
            if (selectionRef.current) drawSelectionOverlay(overlayCtx, selectionRef.current);

            const t = now;

            // --- Grab with open palm (drag stroke) ---
            if (!g.pinch) {
              if (!g.openGrab && openPalm) {
                const pick = pickStroke(palm);
                if (pick) {
                  g.openGrab = true;
                  g.mode = "grab";
                  g.dragStrokeId = pick.id;
                  selectionRef.current = pick.id;
                  g.lastPointer = palm;
                }
              } else if (g.openGrab && !openPalm) {
                g.openGrab = false;
                g.mode = "none";
                g.dragStrokeId = "";
              } else if (g.openGrab && g.dragStrokeId) {
                const dx = palm.x - g.lastPointer.x;
                const dy = palm.y - g.lastPointer.y;
                g.lastPointer = palm;
                translateStroke(g.dragStrokeId, dx, dy);
              }
            }

            // --- Pinch start/end/move ---
            const pinchOn = pinchStrength > 0.7;
            const pinchOff = pinchStrength < 0.35;

            const angle = Math.atan2(index.y - thumb.y, index.x - thumb.x);

            if (!g.pinch && pinchOn) {
              g.pinch = true;
              g.openGrab = false; // pinch has priority
              g.lastPointer = g.smoothed;
              g.lastAngle = angle;

              const pick = pickStroke(g.smoothed);

              if (pick) {
                g.mode = "drag";
                g.dragStrokeId = pick.id;
                selectionRef.current = pick.id;
                g.dragOffset = pick.offset;
              } else {
                g.mode = "draw";
                g.dragStrokeId = "";
                const thickness = clamp(baseThickness + pinchStrength * 5, 2, 20);
                startStroke(g.smoothed, t, thickness);
              }
            } else if (g.pinch && pinchOff) {
              g.pinch = false;
              g.mode = "none";
              g.dragStrokeId = "";
            } else if (g.pinch) {
              // pinch move: draw or manipulate
              const pointerNow = g.smoothed;
              const dx = pointerNow.x - g.lastPointer.x;
              const dy = pointerNow.y - g.lastPointer.y;
              g.lastPointer = pointerNow;

              if (g.mode === "draw") {
                // speed-based thickness (subtle)
                const speed = Math.hypot(dx, dy);
                const thickness = clamp(baseThickness + pinchStrength * 5 + speed * 0.03, 2, 22);
                addStrokePoint(pointerNow, t, thickness);
              } else if (g.dragStrokeId) {
                // drag always
                translateStroke(g.dragStrokeId, dx, dy);

                // rotate when twisting thumb-index angle
                const dAng = angle - g.lastAngle;
                g.lastAngle = angle;

                // deadzone so small noise doesn’t rotate
                if (Math.abs(dAng) > 0.03) {
                  rotateStroke(g.dragStrokeId, dAng);
                  g.mode = "rotate";
                } else if (g.mode === "rotate") {
                  // keep rotate mode sticky if user started rotating
                  if (Math.abs(dAng) < 0.01) g.mode = "drag";
                }
              }
            }
          }

          // subtle HUD
          overlayCtx.save();
          overlayCtx.fillStyle = "rgba(0,0,0,0.25)";
          overlayCtx.fillRect(12, overlay.height - 40, 260, 28);
          overlayCtx.fillStyle = "rgba(255,255,255,0.85)";
          overlayCtx.font = "12px system-ui, -apple-system, Segoe UI";
          overlayCtx.fillText(`FPS ${fps.toFixed(1)} • Infer cap ${inferFpsCap} • Preset ${cameraPreset}`, 20, overlay.height - 22);
          overlayCtx.restore();
        }

        requestAnimationFrame(loop);
      };

      requestAnimationFrame(loop);

      return () => window.removeEventListener("resize", onResize);
    };

    boot().catch((e) => console.error(e));

    return () => {
      mounted = false;
      landmarkerRef.current?.close();
      const stream = videoRef.current?.srcObject as MediaStream | null;
      stream?.getTracks().forEach((t) => t.stop());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showLandmarks, glow, smoothing, baseThickness, brushColor, inferFpsCap, cameraPreset]);

  // Voice toggle effect
  useEffect(() => {
    if (!voiceOn) {
      try {
        recogRef.current?.stop();
      } catch { }
      return;
    }
    if (!recogRef.current) recogRef.current = setupVoice();
    try {
      recogRef.current?.start();
      setVoiceHint("Listening… Say: “change color to red”, “undo”, “clear”, “export”.");
    } catch (e) {
      setVoiceHint("Could not start voice. Try Chrome/Edge and allow mic permission.");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [voiceOn]);

  // When preset changes, restart camera quickly
  useEffect(() => {
    if (!ready) return;
    startCamera(cameraPreset).catch(() => { });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraPreset]);

  const ui = {
    card: {
      background: theme.panel,
      border: `1px solid ${theme.border}`,
      borderRadius: 16,
      padding: 12,
      backdropFilter: "blur(10px)",
    } as React.CSSProperties,
    btn: (primary?: boolean) =>
      ({
        padding: "10px 12px",
        borderRadius: 12,
        border: `1px solid ${primary ? theme.border2 : theme.border}`,
        background: primary ? "rgba(105,240,174,0.18)" : "rgba(0,0,0,0.25)",
        color: theme.fg,
        cursor: "pointer",
        fontWeight: 800,
        transition: "transform 120ms ease, background 120ms ease",
      }) as React.CSSProperties,
    label: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, color: theme.muted, fontSize: 12 } as React.CSSProperties,
  };

  return (
    <div style={{ background: theme.bg, minHeight: "100vh", color: theme.fg, padding: 16 }}>
      <div style={{ maxWidth: 1200, margin: "0 auto", display: "grid", gridTemplateColumns: "320px 1fr", gap: 14 }}>
        {/* Left Panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={ui.card}>
            <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: 0.2 }}>Neon Studio</div>
            <div style={{ fontSize: 12, color: theme.muted, marginTop: 6 }}>
              Pinch = draw / drag / rotate • Open palm = grab stroke • Voice commands
            </div>
          </div>

          <div style={ui.card}>
            <div style={{ fontSize: 13, fontWeight: 900, marginBottom: 10 }}>Tools</div>

            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <button style={ui.btn()} onClick={undo}>Undo</button>
              <button style={ui.btn()} onClick={redo}>Redo</button>
              <button style={ui.btn()} onClick={clearAll}>Clear</button>
              <button style={ui.btn(true)} onClick={exportPng}>Export PNG</button>
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
              <input type="range" min={2} max={20} step={1} value={baseThickness} onChange={(e) => setBaseThickness(parseInt(e.target.value, 10))} />
              <span style={{ width: 32, textAlign: "right" }}>{baseThickness}</span>
            </div>

            <div style={{ height: 10 }} />

            <div style={ui.label}>
              <span>Smoothing</span>
              <input type="range" min={0.2} max={0.9} step={0.05} value={smoothing} onChange={(e) => setSmoothing(parseFloat(e.target.value))} />
              <span style={{ width: 32, textAlign: "right" }}>{smoothing.toFixed(2)}</span>
            </div>

            <div style={{ height: 12 }} />

            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <button style={ui.btn()} onClick={() => setGlow((v) => !v)}>{glow ? "Glow: On" : "Glow: Off"}</button>
              <button style={ui.btn()} onClick={() => setShowLandmarks((v) => !v)}>{showLandmarks ? "Hide Landmarks" : "Show Landmarks"}</button>
            </div>
          </div>

          <div style={ui.card}>
            <div style={{ fontSize: 13, fontWeight: 900, marginBottom: 10 }}>Performance</div>

            <div style={ui.label}>
              <span>Camera preset</span>
              <select
                value={cameraPreset}
                onChange={(e) => setCameraPreset(e.target.value as CameraPreset)}
                style={{ background: "rgba(0,0,0,0.35)", border: `1px solid ${theme.border}`, borderRadius: 10, color: theme.fg, padding: "8px 10px" }}
              >
                <option value="FAST">FAST (lower res)</option>
                <option value="BALANCED">BALANCED</option>
                <option value="HIGH">HIGH (sharper)</option>
              </select>
            </div>

            <div style={{ height: 10 }} />

            <div style={ui.label}>
              <span>Inference FPS cap</span>
              <input type="range" min={10} max={30} step={1} value={inferFpsCap} onChange={(e) => setInferFpsCap(parseInt(e.target.value, 10))} />
              <span style={{ width: 32, textAlign: "right" }}>{inferFpsCap}</span>
            </div>

            <div style={{ marginTop: 10, fontSize: 12, color: theme.muted }}>
              Tip: inference is sync and can block UI; cap FPS + lower res helps. For best responsiveness, move inference to a Worker.
            </div>
          </div>

          <div style={ui.card}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
              <div style={{ fontSize: 13, fontWeight: 900 }}>Voice</div>
              <button style={ui.btn(voiceOn)} onClick={() => setVoiceOn((v) => !v)}>{voiceOn ? "Mic: On" : "Mic: Off"}</button>
            </div>
            <div style={{ marginTop: 10, fontSize: 12, color: theme.muted }}>{voiceHint}</div>
            <div style={{ marginTop: 8, fontSize: 12, color: theme.muted }}>
              Commands: “change color to red”, “undo”, “redo”, “clear”, “export”, “quality fast”.
            </div>
          </div>

          <div style={ui.card}>
            <div style={{ fontSize: 12, color: theme.muted }}>
              Status: {ready ? "Running" : "Loading…"} • FPS {fps.toFixed(1)} • Strokes {strokeCount}
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

          {/* Soft corner fade */}
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