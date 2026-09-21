# Project Plan

**Status**: Integrated
**Created**: 2026-09-21
**Mode**: NEW

---

## 1. Project Overview

**Goal**: Build DeepGuard, a local-first multimodal synthetic-content detection dashboard for text, audio, image, and video, with transparent heuristic/mock risk scoring and explainability evidence. The project is designed so that every module is independently testable.

**App Type**: SPA + API

**API Login**: No

**Mode**: NEW

**Deployment Plan**: No deployment plan found

---

## 2. Detection API — backend

| Component | Technology |
|-----------|-----------|
| **Language** | Python |
| **Runtime** | CPython |
| **Package Manager** | pip |
| **Test Runner** | pytest |
| **Mocking Library** | unittest.mock |
| **Test Command** | pytest |
| **Orchestration** | docker-compose |

The FastAPI API exposes health/version, text analysis, and multipart media analysis endpoints. Detector implementations follow a replaceable interface and return deterministic risk score, confidence, label, modality-specific evidence, and detector metadata. Audio, image, and video use lightweight metadata/content heuristics with temporary local files; no heavyweight production models are required for the MVP.

---

## 3. DeepGuard Dashboard — frontend

| Component | Technology |
|-----------|-----------|
| **Language** | TypeScript |
| **Framework** | React + Vite |
| **Package Manager** | npm |
| **Test Runner** | vitest |
| **Mocking Library** | vi.mock |
| **Test Command** | npm test |

---

## 4. Services Required

| Azure Service | Role in App | Environment Variable | Default Value (Local) | Classification |
|---------------|------------|---------------------|----------------------|----------------|
| SQLite (local) | Persist analysis metadata, detector outputs, and current-session history locally | `DEEPGUARD_DB_PATH` | `./data/deepguard.db` | Essential |
| Local filesystem | Hold uploaded media and generated evidence temporarily during analysis | `DEEPGUARD_UPLOAD_DIR` | `./data/uploads` | Essential |
| No Azure services | Azure deployment is explicitly out of scope for this local-first MVP | — | — | Deferred |

---

## 5. Prerequisites

### Run

| Tool | Service(s) | Installed | Version |
|------|------------|-----------|---------|
| Node.js | DeepGuard Dashboard | ✅ | 24.15.0 |
| npm | DeepGuard Dashboard | ✅ | bundled with Node.js 24.15.0 |
| Python | Detection API | ✅ | 3.14.4150 |
| pip | Detection API | ✅ | available at `C:\Python314\Scripts\pip.exe` |
| SQLite | Detection API | ❓ | Python `sqlite3` module is the fallback |

### Debug

| Tool | Service(s) | Installed | Version |
|------|------------|-----------|---------|
| Docker | Detection API, DeepGuard Dashboard | ❓ | not confirmed |
| Docker Compose | Detection API, DeepGuard Dashboard | ❓ | not confirmed |
| VS Code Python extension | Detection API | ❓ | not scanned |
| VS Code ESLint extension | DeepGuard Dashboard | ❓ | not scanned |


## 6. Design System & UI

**Component Library**: Fluent UI v9
**Style Direction**: A calm forensic console with crisp teal actions, amber risk signals, and generous evidence panels. The interface should feel analytical and trustworthy, using restrained surfaces, compact metadata, and clear separation between heuristic MVP output and future model-backed findings.
**Typography**: IBM Plex Sans, "Segoe UI", sans-serif

### Color Palette

| Token | Hex | Usage |
|-------|-----|-------|
| `primary` | `#0B6E69` | Analyze actions, active navigation, detector status accents |
| `accent`  | `#D97706` | Risk markers, confidence highlights, and attention states |
| `surface` | `#F4F7F6` | Main dashboard background and analysis workspace |
| `text`    | `#17211F` | Findings, labels, scores, and body copy |
| `muted`   | `#687773` | Evidence captions, timestamps, helper text, and secondary metadata |
| `border`  | `#D5E0DD` | Panel boundaries, upload zones, table dividers, and form controls |

### Pages

| Page | Route | Purpose | Layout |
|------|-------|---------|--------|
| Analyze | `/` | Submit text or media and inspect the latest heuristic finding | `header, nav, main, hero, form, grid, card-list, split(form|card-list)` |
| Analysis History | `/history` | Review local-session analyses and filter by modality or risk | `header, nav, main, table, actions, footer` |
| Analysis Detail | `/analysis/demo-1042` | Inspect one finding's score, evidence, and detector metadata | `header, nav, main, two-column(card-list+card-list), action-bar` |

### Sample Content

Analyze — latest findings:
| Modality | Input | Risk score | Confidence | State |
|----------|-------|------------|------------|-------|
| Text | "The committee approved the revised climate brief." | 18 / 100 | 0.91 | Likely authentic |
| Image | `press-room-04.png` | 74 / 100 | 0.68 | Review recommended |
| Audio | `briefing-2026-09-18.wav` | 52 / 100 | 0.57 | Uncertain |

Analysis History — local analysis:
| ID | Modality | Label | Score | Timestamp |
|----|----------|-------|-------|-----------|
| `demo-1042` | Image | Review recommended | 74 | 2026-09-21 10:42 |
| `demo-1041` | Text | Likely authentic | 18 | 2026-09-21 10:36 |
| `demo-1039` | Audio | Uncertain | 52 | 2026-09-21 09:58 |
| `demo-1037` | Video | Likely authentic | 29 | 2026-09-20 17:21 |

Analysis Detail — demo-1042:
| Field | Value |
|-------|-------|
| File | `press-room-04.png` |
| Modality | Image |
| Risk score | 74 / 100 |
| Confidence | 0.68 |
| Detector | `image-heuristic-v1` |
| Evidence | Edge-density hotspot around face boundary; recompression variance detected |

---

## 7. Project Structure

```
DeepGuard/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── features/analysis/
│   │   ├── pages/
│   │   ├── api/
│   │   ├── types/
│   │   └── main.tsx
│   ├── public/
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
├── backend/
│   ├── app/
│   │   ├── api/routes/
│   │   ├── detectors/
│   │   ├── models/
│   │   ├── services/
│   │   ├── db/
│   │   └── main.py
│   ├── tests/
│   ├── pyproject.toml
│   └── requirements.txt
├── data/
│   ├── uploads/.gitkeep
│   └── .gitkeep
└── docker-compose.yml
```

---

## 8. Route Definitions

| # | Method | Path | Description | Request Body | Response Body | Status Codes |
|---|--------|------|-------------|--------------|--------------|--------------|
| 1 | GET | `/api/health` | Health check for API and local SQLite | — | `{ status, services }` | 200, 503 |
| 2 | GET | `/api/version` | Return API and detector contract versions | — | `{ apiVersion, detectorContractVersion }` | 200 |
| 3 | POST | `/api/analyze/text` | Analyze text with the deterministic text detector | `{ text }` | `{ analysisId, modality, label, riskScore, confidence, evidence, detector }` | 200, 400, 422 |
| 4 | POST | `/api/analyze/media` | Analyze audio, image, or video upload with a modality detector | multipart `{ file, modality }` | `{ analysisId, modality, label, riskScore, confidence, evidence, detector }` | 200, 400, 413, 422 |
| 5 | GET | `/api/analyses` | List local analysis history with optional modality/risk filters | query `modality`, `risk` | `{ items, total }` | 200 |
| 6 | GET | `/api/analyses/{analysisId}` | Retrieve one analysis and its evidence | — | `{ analysisId, input, result, evidence, detector, createdAt }` | 200, 404 |

---

## 9. Next Steps

1. Run **azure-project-scaffold** to execute this plan
2. Run **azure-project-integrate** to wire the frontend to live data, smoke-test the backend, and create the migrations
3. Run **azure-debug-plan** → **azure-debug-generate** for Docker emulators and VS Code debugging
4. Run the **azure-deploy** agent when ready; Azure deployment remains out of scope for this MVP
