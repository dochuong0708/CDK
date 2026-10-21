ALTER TABLE analyses ADD COLUMN style_risk INTEGER CHECK (style_risk BETWEEN 0 AND 100);
ALTER TABLE analyses ADD COLUMN style_label TEXT;
ALTER TABLE analyses ADD COLUMN fraud_risk INTEGER CHECK (fraud_risk BETWEEN 0 AND 100);
ALTER TABLE analyses ADD COLUMN factuality_risk INTEGER CHECK (factuality_risk BETWEEN 0 AND 100);
ALTER TABLE analyses ADD COLUMN factuality_label TEXT;
ALTER TABLE analyses ADD COLUMN fact_check_id TEXT;

CREATE TABLE IF NOT EXISTS fact_checks (
    report_id TEXT PRIMARY KEY,
    input TEXT NOT NULL,
    summary TEXT NOT NULL,
    factuality_risk REAL CHECK (factuality_risk BETWEEN 0 AND 1),
    factuality_confidence REAL NOT NULL DEFAULT 0,
    retrieval_backend TEXT,
    report_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fact_checks_created_at ON fact_checks (created_at DESC);
