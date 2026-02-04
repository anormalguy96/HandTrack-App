import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [fps, setFps] = useState(0);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let mounted = true;
    let handLandmarker: HandLandmarker | null = null;
    let last = performance.now();
    let ema = 0;

    const start = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm"
      );

      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        },
        runningMode: "VIDEO",
        numHands: 1,
      });

      setReady(true);

      const loop = async () => {
        if (!mounted || !videoRef.current || !canvasRef.current || !handLandmarker) return;

        const now = performance.now();
        const dt = Math.max(1, now - last);
        const inst = 1000 / dt;
        ema = ema === 0 ? inst : (ema * 0.9 + inst * 0.1);
        last = now;
        setFps(ema);

        const video = videoRef.current;
        const canvas = canvasRef.current;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const res = handLandmarker.detectForVideo(video, now);

        const ctx = canvas.getContext("2d")!;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = "rgba(105,240,174,0.9)";
        for (const hand of res.landmarks ?? []) {
          for (const p of hand) {
            ctx.beginPath();
            ctx.arc(p.x * canvas.width, p.y * canvas.height, 5, 0, Math.PI * 2);
            ctx.fill();
          }
        }

        requestAnimationFrame(loop);
      };

      requestAnimationFrame(loop);
    };

    start().catch(console.error);

    return () => {
      mounted = false;
      handLandmarker?.close();
      const stream = videoRef.current?.srcObject as MediaStream | null;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return (
    <div style={{ padding: 16, color: "#fff", background: "#0B0D10", minHeight: "100vh" }}>
      <h2 style={{ margin: 0, marginBottom: 8 }}>Box-of-Scraps — Web Hand Tracking</h2>
      <div style={{ opacity: 0.8, marginBottom: 12 }}>
        {ready ? `Running • FPS ${fps.toFixed(1)}` : "Loading model..."}
      </div>

      <div style={{ position: "relative", width: "min(900px, 100%)", aspectRatio: "16/9", border: "1px solid rgba(255,255,255,0.12)" }}>
        <video ref={videoRef} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover", transform: "scaleX(-1)" }} />
        <canvas ref={canvasRef} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none", transform: "scaleX(-1)" }} />
      </div>
    </div>
  );
}