CREATE TABLE IF NOT EXISTS user (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  nickname   TEXT    NOT NULL,
  embedding  TEXT,               -- JSON array of 128 floats, NULL until recorded
  is_solo    INTEGER NOT NULL DEFAULT 0,
  age        INTEGER,
  gender     TEXT
);

CREATE TABLE IF NOT EXISTS session (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  user1_id   INTEGER NOT NULL,
  user2_id   INTEGER NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user1_id) REFERENCES user (id),
  FOREIGN KEY (user2_id) REFERENCES user (id)
);

CREATE TABLE IF NOT EXISTS result (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL,
  score      INTEGER NOT NULL,
  similarity REAL,               -- raw cosine similarity [-1, 1]
  data_path  TEXT,
  FOREIGN KEY (session_id) REFERENCES result (id)
);
