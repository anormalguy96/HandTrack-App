export type DeviceTier = "low" | "mid" | "high";

export type QualityProfile = {
  inputWidth: number;
  inputHeight: number;
  roiEnabled: boolean;
  frameSkip: number; // 0 = no skip, 1 = skip every other frame, etc.
  smoothing: number; // 0..1
};

const PROFILES: Record<DeviceTier, QualityProfile> = {
  low:  { inputWidth: 320, inputHeight: 240, roiEnabled: true,  frameSkip: 1, smoothing: 0.85 },
  mid:  { inputWidth: 480, inputHeight: 360, roiEnabled: true,  frameSkip: 0, smoothing: 0.65 },
  high: { inputWidth: 640, inputHeight: 480, roiEnabled: false, frameSkip: 0, smoothing: 0.45 }
};

export function pickQualityProfile(opts: {
  tier: DeviceTier;
  thermalThrottle?: boolean;
}): QualityProfile {
  const base = PROFILES[opts.tier];
  if (!opts.thermalThrottle) return base;

  // If throttled, degrade safely.
  return {
    ...base,
    inputWidth: Math.max(240, Math.floor(base.inputWidth * 0.75)),
    inputHeight: Math.max(180, Math.floor(base.inputHeight * 0.75)),
    frameSkip: Math.max(base.frameSkip, 1),
    smoothing: Math.min(0.95, base.smoothing + 0.1)
  };
}
