import { useEffect, useState } from "react";
import { startFpsMeter } from "../../../benchmarks/web/raf_fps";

const API = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8080";

export default function App() {
  const [fps, setFps] = useState<number>(0);
  const [health, setHealth] = useState<any>(null);

  useEffect(() => startFpsMeter(setFps), []);

  async function loadHealth() {
    const res = await fetch(`${API}/api/health`);
    setHealth(await res.json());
  }

  return (
    <div style={{ fontFamily: "system-ui", padding: 24, lineHeight: 1.4 }}>
      <h1 style={{ margin: 0 }}>HandTrack Web</h1>
      <p style={{ marginTop: 8, opacity: 0.8 }}>
        Starter dashboard (FPS meter + API connectivity). Replace with admin console / demo hosting.
      </p>

      <div style={{ display: "flex", gap: 12, marginTop: 16, flexWrap: "wrap" }}>
        <div style={{ border: "1px solid rgba(0,0,0,0.12)", borderRadius: 14, padding: 12, minWidth: 220 }}>
          <div style={{ fontSize: 12, opacity: 0.7 }}>FPS</div>
          <div style={{ fontSize: 28, fontWeight: 700 }}>{fps}</div>
        </div>

        <div style={{ border: "1px solid rgba(0,0,0,0.12)", borderRadius: 14, padding: 12, minWidth: 320 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ fontSize: 12, opacity: 0.7 }}>API</div>
              <div style={{ fontWeight: 700 }}>{API}</div>
            </div>
            <button onClick={loadHealth} style={{ padding: "8px 10px", borderRadius: 12 }}>
              Test /api/health
            </button>
          </div>
          <pre style={{ marginTop: 10, fontSize: 12, opacity: 0.85, whiteSpace: "pre-wrap" }}>
            {health ? JSON.stringify(health, null, 2) : "Click to fetch..."}
          </pre>
        </div>
      </div>

      <hr style={{ margin: "22px 0", opacity: 0.25 }} />

      <h2 style={{ margin: 0 }}>Next</h2>
      <ul>
        <li>Integrate MediaPipe/WebAssembly demo pipeline (web)</li>
        <li>Device profile viewer (FPS, throttling, tier)</li>
        <li>Admin console (feature flags, placements, export quotas)</li>
      </ul>
    </div>
  );
}
