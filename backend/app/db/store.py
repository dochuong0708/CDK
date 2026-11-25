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
        self.connection.execute('CREATE TABLE IF NOT EXISTS audio_details (analysis_id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        self.connection.commit()

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
        audio_payload = item.model_dump(
            include={'audio', 'audioScore', 'audioThreshold', 'audioLatencyMs', 'audioXai'},
            exclude_none=True,
        )
        if audio_payload:
            self.connection.execute(
                'INSERT INTO audio_details VALUES (?, ?)',
                (item.analysisId, json.dumps(audio_payload, ensure_ascii=False)),
            )
        self.connection.commit()
        return item

    def list(self, modality: str | None = None, risk: int | None = None) -> list[Analysis]:
        query = (
            'SELECT analyses.*, audio_details.payload FROM analyses '
            'LEFT JOIN audio_details ON analyses.analysis_id = audio_details.analysis_id '
            'WHERE 1 = 1'
        )
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
            'SELECT analyses.*, audio_details.payload FROM analyses '
            'LEFT JOIN audio_details ON analyses.analysis_id = audio_details.analysis_id '
            'WHERE analyses.analysis_id = ?', (analysis_id,)
        ).fetchone()
        if not row:
            return None
        return self._from_row(row)

    @staticmethod
    def _from_row(row: tuple) -> Analysis:
        item = Analysis(
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
        if len(row) > 9 and row[9]:
            return Analysis.model_validate({**item.model_dump(), **json.loads(row[9])})
        return item
