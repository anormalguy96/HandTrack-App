import express from "express";

const app = express();
app.use(express.json());

/**
 * Enterprise-friendly ads config:
 * - Server decides if ads are enabled (by tier/region/consent)
 * - App never hardcodes ad unit IDs in repo
 */
app.get("/v1/ads/config", (req, res) => {
  const region = (req.query.region || "unknown").toString().toLowerCase();
  const tier = (req.query.tier || "free").toString().toLowerCase(); // free | premium | enterprise
  const consent = (req.query.consent || "false").toString().toLowerCase() === "true";

  const enabled = tier === "free" && consent && region !== "blocked";
  return res.json({
    enabled,
    provider: enabled ? "admob" : "none",
    // Put REAL IDs via env in prod (never commit real IDs)
    bannerUnitId: enabled ? (process.env.ADMOB_BANNER_ID || "test-banner") : null,
    interstitialUnitId: enabled ? (process.env.ADMOB_INTERSTITIAL_ID || "test-interstitial") : null,
    frequencyCaps: {
      interstitialPerMinute: 1,
    },
    privacy: {
      onDeviceOnly: true,
      noCameraFramesUploaded: true,
    }
  });
});

app.get("/health", (_req, res) => res.json({ ok: true }));

const port = process.env.PORT || 8787;
app.listen(port, () => console.log(`ads-service running on :${port}`));