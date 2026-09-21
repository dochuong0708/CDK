# DeepGuard Integration Handoff

## Backend
- Folder: `backend/`
- Run: `uvicorn app.main:app --reload --port 8000` (from `backend/`)
- Build/check: `python -m compileall app`; tests: `python -m pytest tests`
- Health: `GET /api/health`

## Frontend
- Folder: `frontend/`
- Build: `npm run build`
- Dev: `npm run dev -- --host 0.0.0.0`
- API seam: `frontend/src/api/index.ts` (swap `mockClient` for live client)
- Delete after wiring: `frontend/src/api/mockClient.ts`, `frontend/src/api/previewState.ts`, any `frontend/src/mocks/*`, and locally duplicated types.

## API routes
- GET `/api/health`
- GET `/api/version`
- POST `/api/analyze/text`
- POST `/api/analyze/media`
- GET `/api/analyses`
- GET `/api/analyses/{analysisId}`

## Database
- Type: SQLite
- Migration tool/directory: integrate agent to choose and create under `backend/migrations/`
- Connection env: `DEEPGUARD_DB_PATH` (default `./data/deepguard.db`)
- Upload env: `DEEPGUARD_UPLOAD_DIR` (default `./data/uploads`)
- NO seed data is to be created. Preserve only runtime test fixtures where needed.
- Persist analysis metadata, detector outputs, evidence, input name, modality, and timestamps.

## Shared types
- Current frontend contracts: `frontend/src/types/index.ts`.
- Backend contracts: `backend/app/models/schemas.py`.
- Align the live client method-for-method with the frontend `ApiClient` interface.

## Services
- Essential: local SQLite, local filesystem uploads.
- Enhancement: none; Azure services are out of scope for this MVP.

## Integration results

- Created and applied `backend/migrations/001_create_analyses.sql`; the SQLite database contains `analyses` and `schema_migrations` only, with no seed rows added.
- Replaced the in-memory analysis store with SQLite persistence and verified the migration ledger applies cleanly.
- Smoke-tested every route over HTTP: health, version, text analysis, media analysis, analysis list, and analysis detail all returned successfully.
- Added the typed live client at `frontend/src/api/client.ts`, switched `frontend/src/api/index.ts` to it, and configured the Vite `/api` proxy for port 8000.
- Removed `frontend/src/api/mockClient.ts` and `frontend/src/api/previewState.ts`; no mock references remain under `frontend/src`.
- Frontend production build passed, and the running dashboard rendered the two persisted smoke-test analyses through the frontend-to-backend proxy.
