# Absolute Builders – AI-Powered Estimation Suite

CAD Material Tracker: upload tag lists and PDF drawings, run AI-powered extraction, and download results as CSV. Built for Absolute Builders.

## Stack

- **Backend:** FastAPI (Python 3.12), Postgres (Supabase), JWT auth, WebSocket for live progress
- **Frontend:** Next.js 16 (ported to Next.js), React 19, Tailwind CSS
- **Deploy:** Backend on EC2 via systemd; frontend in Docker; Nginx as single entry (only port 80 exposed)

---

## Local development

### Backend

```bash
# From repo root
python -m venv venv312
source venv312/bin/activate   # Windows: venv312\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` in the repo root (see [Environment variables](#environment-variables)).

```bash
# Run backend (port 8000)
python run.py

# With auto-reload
python run.py --reload
```

Optional: run migrations if you use Alembic:

```bash
alembic upgrade head
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The app will call the backend at `http://localhost:8000` in dev.

---

## Environment variables

**Backend** (repo root `.env`; never commit this file):

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | Postgres connection (e.g. Supabase pooler, port 6543) |
| `DIRECT_URL` | Optional | Direct Postgres URL for migrations (port 5432) |
| `JWT_SECRET_KEY` | Yes | Long random secret for JWT (e.g. 32+ chars) |
| `OPENAI_API_KEY` | Optional | For sheet-name extraction |
| `ROBOFLOW_API_KEY` | Optional | For drawing detection |
| `ROBOFLOW_MODEL_ID` | Optional | Roboflow model id |
| `LLAMA_PARSE_API_KEY` | Optional | For PDF parsing |
| `AUTO_CLEANUP` | Optional | `true` to auto-delete temp files |
| `ALLOWED_ORIGINS` | Optional | Comma-separated CORS origins (production) |

**Frontend:** In production the browser uses the same origin; Nginx proxies API/WS to the backend. No frontend `.env` needed for API URL.

---

## Deployment (EC2)

Push to `main` to trigger the GitHub Actions workflow. It will:

1. SSH to EC2 and install/configure system deps (Python, Tesseract, Nginx, Docker, etc.)
2. Clone/pull the repo and set up the backend (venv, `.env` from secrets, systemd service running `run.py`)
3. Build and run the frontend in Docker (`docker compose up -d frontend`)
4. Configure Nginx so only **port 80** is the entry: `/` → frontend (Docker :3000), `/auth`, `/upload`, `/ws`, etc. → backend (localhost:8000)

**Required GitHub Secrets:**

- `EC2_HOST` – EC2 hostname or IP  
- `EC2_SSH_KEY` – Private key for `ec2-user`  
- `DATABASE_URL` – Postgres URL (no quotes in secret)  
- `JWT_SECRET_KEY` – Strong random secret  
- `ROBOFLOW_API_KEY`, `ROBOFLOW_MODEL_ID`  
- `OPENAI_API_KEY`, `LLAMA_PARSE_API_KEY`  

**EC2 security group:** Open only **22** (SSH) and **80** (HTTP). Do not open 3000 or 8000.

After deploy, the app is at `http://<EC2_HOST>`.

---

## Project layout

```
├── app/                    # FastAPI backend
│   ├── api/                # Routes (auth, upload, jobs, cleanup, websocket)
│   ├── core/               # Config, security (JWT, password hashing)
│   ├── db/                 # DB session
│   ├── models/             # SQLAlchemy models
│   ├── schemas/            # Pydantic schemas
│   ├── services/           # PDF processing, AI, cleanup, job manager
│   └── utils/              # Security (path sanitization, etc.)
├── frontend/               # Next.js app
│   ├── app/                # Pages (login, dashboard, docs, profile)
│   ├── components/        # UI (Header, UploadPanel, ProcessingView, etc.)
│   ├── hooks/              # useWebSocket
│   └── public/             # Static assets, sample_input_sheet.xlsx
├── alembic/                # DB migrations
├── templates/              # Backend HTML (legacy)
├── run.py                  # Single entry point (dev + production)
├── docker-compose.yml      # Frontend container
├── requirements.txt
└── .github/workflows/      # Deploy to EC2
```

---

## User flow

1. **Login / Sign up** – JWT stored in browser; backend uses Postgres.
2. **Step 1:** Upload a tag list (CSV or Excel) with a **Tags** column (and optional **Material Type**). See **Documentation** in the app for format and a sample sheet.
3. **Step 2:** Upload a CAD PDF, set optional page range, then **Start processing**. Progress streams over WebSocket; detected drawings and tags appear live.
4. **Results:** View and download CSV (total tags and per-tag counts on the results page).

---

## Security

- Auth: JWT, protected routes and WebSocket.
- CORS: Controlled via `ALLOWED_ORIGINS` in production.
- Uploads: Filenames sanitized; WebSocket `pdf_path` validated to stay under upload dir.
- See `SECURITY.md` for a production checklist (HTTPS, JWT secret, rate limiting, etc.).

---

## License

Proprietary – © Absolute Builders.
