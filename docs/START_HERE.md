# Start Here

## Extract & open in VS Code
1. Unzip the folder
2. Open the root folder in VS Code

## Create GitHub repo
```bash
git init
git add .
git commit -m "chore: initial enterprise starter scaffold"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## Run order
- `infra/` → docker compose up
- `services/api/` → run Spring Boot
- `apps/web/` → run Vite
- `apps/mobile/` → run Flutter
