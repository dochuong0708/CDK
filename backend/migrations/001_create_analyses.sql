CREATE TABLE IF NOT EXISTS analyses (
    analysis_id TEXT PRIMARY KEY,
    modality TEXT NOT NULL CHECK (modality IN ('Text', 'Image', 'Audio', 'Video')),
    label TEXT NOT NULL,
    risk_score INTEGER NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    evidence_json TEXT NOT NULL,
    detector TEXT NOT NULL,
    input TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analyses_modality ON analyses (modality);
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses (created_at DESC);