import sqlite3
from pathlib import Path


def apply_migrations(connection: sqlite3.Connection, directory: Path) -> None:
    connection.execute(
        'CREATE TABLE IF NOT EXISTS schema_migrations ('
        'version TEXT PRIMARY KEY, applied_at TEXT NOT NULL)'
    )
    applied = {
        row[0] for row in connection.execute('SELECT version FROM schema_migrations')
    }
    for migration in sorted(directory.glob('*.sql')):
        if migration.name in applied:
            continue
        connection.executescript(migration.read_text(encoding='utf-8'))
        connection.execute(
            'INSERT INTO schema_migrations (version, applied_at) '
            "VALUES (?, datetime('now'))",
            (migration.name,),
        )
    connection.commit()