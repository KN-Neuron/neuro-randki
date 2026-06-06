import sqlite3
from flask import current_app, g


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    db = get_db()
    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))
    # Migration: add columns that may be missing in older DBs
    for stmt in [
        'ALTER TABLE user ADD COLUMN embedding TEXT',
        'ALTER TABLE user ADD COLUMN is_solo INTEGER NOT NULL DEFAULT 0',
        'ALTER TABLE result ADD COLUMN similarity REAL',
    ]:
        try:
            db.execute(stmt)
            db.commit()
        except Exception:
            pass  # column already exists


def init_app(app):
    app.teardown_appcontext(close_db)
