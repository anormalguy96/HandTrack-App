# HandTrack Enterprise Suite (Starter Monorepo)

A production-oriented scaffold to evolve **Box-of-Scraps** hand-gesture demos into an **enterprise-grade**, **privacy-first**, **multi-platform** product.

**Includes (scaffolded):**
- **Design system** (tokens + guidelines)
- **Mobile app** (Flutter starter)
- **Web app** (React + Vite starter)
- **Backend** (Spring Boot API)
- **Ads service** (optional, privacy-aware)
- **ML/R&D** (Python package + pipeline starter)
- **Efficiency & benchmarking** (adaptive quality + FPS tools)
- **Infra** (docker-compose, env examples, deployment docs)

> This repo is a **starter**: structure + boundaries + docs + runnable minimal apps.
> The real-time on-device hand-tracking engine is stubbed and documented under `engine/`.

---

## Quick start (local)

### 1) Prereqs
- Docker Desktop
- Java 17+ + Maven
- Node 20+
- Flutter SDK
- Python 3.11+

### 2) Start infra (DB + Redis + ads-service)
```bash
cd infra
cp ../.env.example ../.env
docker compose up -d
```

### 3) Start backend API
```bash
cd services/api
mvn spring-boot:run
```
API: `http://localhost:8080/api/health`

### 4) Start web app
```bash
cd apps/web
npm i
npm run dev
```

### 5) Mobile app
```bash
cd apps/mobile
flutter pub get
flutter run
```

---

## Monorepo map
- `design/` — tokens + component guidelines
- `apps/mobile/` — Flutter app starter
- `apps/web/` — React/Vite starter
- `services/api/` — Spring Boot API starter
- `services/ads-service/` — optional ads config service
- `ml/` — Python R&D starter
- `packages/quality/` — adaptive quality selector
- `benchmarks/` — FPS/latency helpers
- `infra/` — docker compose + local infra docs
- `docs/` — architecture, security, privacy, performance, ads, deployment

---

## Key principles
- **Privacy-first:** camera frames stay on-device by default.
- **Performance-first:** adaptive quality + device profiling.
- **Enterprise-ready:** docs + modular services + safe defaults.
- **No secrets in git:** never commit `.env`.
