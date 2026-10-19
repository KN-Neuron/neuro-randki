CREATE TABLE IF NOT EXISTS user (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  nickname   TEXT    NOT NULL,
  embedding  TEXT,               -- JSON array of floats, NULL until recorded
  is_solo    INTEGER NOT NULL DEFAULT 0,
  age        INTEGER,
  gender     TEXT,
  archetype  TEXT,               -- zen, focus, visionary, genius, thinker
  band_powers TEXT,
  signal_path TEXT,
  stim_signature TEXT,          -- JSON: trial-averaged response per colour (shared stimulus)
  stim_quality REAL             -- split-half reliability of that signature
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
  similarity REAL,               -- raw similarity score
  breakdown  TEXT,               -- JSON with harmony, energy, cognition, diagnosis
  data_path  TEXT,
  FOREIGN KEY (session_id) REFERENCES session (id)
);
