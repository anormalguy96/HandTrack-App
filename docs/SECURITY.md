# Security

## Rules
- Never commit secrets (`.env` not in git)
- HTTPS everywhere in production
- Mobile: store tokens in secure storage
- Web: refresh token via HttpOnly cookie (future)

## TODOs
- rate limiting
- audit logging (admin)
- signed export URLs
