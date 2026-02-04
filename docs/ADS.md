# Ads (Optional Module)

## Goals
- Ads never depend on camera/biometric data.
- Consent-first.
- Ads removable via premium tier.

## Architecture
`services/ads-service` returns:
- provider config (AdMob, etc.)
- placements enabled/disabled
- frequency caps
- region gating (future)
