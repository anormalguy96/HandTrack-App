# Deployment

## Local
Use `infra/docker-compose.yml`.

## Prod (suggested)
- API: container (Kubernetes/ECS/Fly.io)
- DB: managed Postgres
- Redis: managed
- Web: Netlify/Vercel
- Observability: OpenTelemetry exporter + dashboard
