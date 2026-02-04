import express from "express";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json());

// Privacy-first: ads config only. Never accept camera/biometric data here.
app.get("/ads/config", (req, res) => {
  const provider = process.env.ADS_PROVIDER ?? "admob";
  res.json({
    provider,
    enabled: true,
    placements: {
      bannerHome: true,
      interstitialExport: false,
      rewardedProTrial: true
    },
    frequencyCaps: { interstitialPerHour: 2 },
    note: "Starter config. Add consent + regional gating before production."
  });
});

app.get("/health", (req, res) => res.json({ ok: true }));

const port = 8090;
app.listen(port, () => console.log(`ads-service listening on :${port}`));
