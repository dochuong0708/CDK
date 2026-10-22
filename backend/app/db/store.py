from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from app.db.migrate import apply_migrations
from app.models.fact_checking import FactCheckReport
from app.models.schemas import Analysis


class AnalysisStore:
    def __init__(self, path: str = './data/deepguard.db'):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        apply_migrations(self.connection, Path(__file__).parents[2] / 'migrations')

    def add(self, item: Analysis) -> Analysis:
        self.connection.execute(
            'INSERT INTO analyses '
            '(analysis_id, modality, label, risk_score, confidence, evidence_json, detector, input, '
            'created_at, style_risk, style_label, fraud_risk, factuality_risk, factuality_label, '
            'fact_check_id) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
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
                item.styleRisk,
                item.styleLabel,
                item.fraudRisk,
                item.factualityRisk,
                item.factualityLabel,
                item.factCheckId,
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

    def add_fact_check(self, report: FactCheckReport, input_text: str) -> FactCheckReport:
        saved = report.model_copy(
            update={
                'reportId': report.reportId or f'fc-{uuid4().hex[:8]}',
                'createdAt': report.createdAt or datetime.now(),
            }
        )
        self.connection.execute(
            'INSERT INTO fact_checks '
            '(report_id, input, summary, factuality_risk, factuality_confidence, '
            'retrieval_backend, report_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (
                saved.reportId,
                input_text,
                saved.summary,
                saved.factuality_risk,
                saved.factuality_confidence,
                saved.retrieval_backend,
                saved.model_dump_json(),
                saved.createdAt.isoformat(),
            ),
        )
        self.connection.commit()
        return saved

    def list_fact_checks(self, limit: int = 50) -> list[FactCheckReport]:
        rows = self.connection.execute(
            'SELECT report_json FROM fact_checks ORDER BY created_at DESC LIMIT ?', (limit,)
        )
        return [FactCheckReport.model_validate_json(row['report_json']) for row in rows]

    def get_fact_check(self, report_id: str) -> FactCheckReport | None:
        row = self.connection.execute(
            'SELECT report_json FROM fact_checks WHERE report_id = ?', (report_id,)
        ).fetchone()
        return FactCheckReport.model_validate_json(row['report_json']) if row else None

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Analysis:
        return Analysis(
            analysisId=row['analysis_id'],
            modality=row['modality'],
            label=row['label'],
            riskScore=row['risk_score'],
            confidence=row['confidence'],
            evidence=json.loads(row['evidence_json']),
            detector=row['detector'],
            input=row['input'],
            createdAt=row['created_at'],
            styleRisk=row['style_risk'],
            styleLabel=row['style_label'],
            fraudRisk=row['fraud_risk'],
            factualityRisk=row['factuality_risk'],
            factualityLabel=row['factuality_label'],
            factCheckId=row['fact_check_id'],
        )
