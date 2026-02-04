# API (Spring Boot)

Starter endpoints:
- `GET /api/health` (public)
- `GET /api/config` (Basic-auth)
- `GET /api/ads/config` (Basic-auth, proxies ads-service)

Run:
```bash
mvn spring-boot:run
```

Auth note:
Replace Basic auth with:
- Access JWT
- Refresh token strategy (HttpOnly cookie for web, secure storage for mobile)
