import json
import sqlite3
from pathlib import Path

from app.db.migrate import apply_migrations
from app.models.schemas import Analysis


class AnalysisStore:
    def __init__(self, path: str = './data/deepguard.db'):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        apply_migrations(self.connection, Path(__file__).parents[2] / 'migrations')

    def add(self, item: Analysis) -> Analysis:
        self.connection.execute(
            'INSERT INTO analyses '
            '(analysis_id, modality, label, risk_score, confidence, evidence_json, detector, input, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                item.analysisId,
                item.modality,
                item.label,
                item.riskScore,
                item.confidence,
                json.dumps(item.evidence),
                item.detector,
                item.input,
                item.createdAt.isoformat(),
            ),
        )
        self.connection.commit()
        return item

    def list(self, modality: str | None = None, risk: int | None = None) -> list[Analysis]:
        query = 'SELECT * FROM analyses WHERE 1 = 1'
        parameters: list[str | int] = []
        if modality:
            query += ' AND modality = ?'
            parameters.append(modality)
        if risk is not None:
            query += ' AND risk_score >= ?'
            parameters.append(risk)
        query += ' ORDER BY created_at DESC'
        return [self._from_row(row) for row in self.connection.execute(query, parameters)]

    def get(self, analysis_id: str) -> Analysis | None:
        row = self.connection.execute(
            'SELECT * FROM analyses WHERE analysis_id = ?', (analysis_id,)
        ).fetchone()
        return self._from_row(row) if row else None

    @staticmethod
    def _from_row(row: tuple) -> Analysis:
        return Analysis(
            analysisId=row[0],
            modality=row[1],
            label=row[2],
            riskScore=row[3],
            confidence=row[4],
            evidence=json.loads(row[5]),
            detector=row[6],
            input=row[7],
            createdAt=row[8],
        )
