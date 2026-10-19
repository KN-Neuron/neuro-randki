#!/usr/bin/env python3
"""
Seed realistic event participants from data/*.fif recordings or mock data.
Populates SQLite user, session, and result tables so the friendship graph
and leaderboard are immediately alive with real brainwave connections.
"""
import glob
import json
import os
import random
import sys
import numpy as np

# Ensure app root is on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import get_db, init_db
from app import app, RECORDINGS_DIR, _save_embedding, _refresh_pair_results
import shared_stimulus as stim
import model as model_mod

DEMO_NICKS = [
    ("CyberAnia", False),
    ("NeuroTomek", False),
    ("SynapseOla", False),
    ("CortexMichał", False),
    ("AlphaKamil", True),
    ("ThetaZuzia", False),
    ("BetaPiotr", False),
    ("DeltaAdam", True),
    ("GammaMagda", False),
    ("MatrixJan", False),
    ("CyberWojtek", True),
    ("BrainNatalia", False),
    ("ChopinEEG", False),
    ("QuantumBartek", True),
    ("NeuronMarta", False),
    ("SynapticKuba", False),
    ("PulseKarolina", True),
    ("BioHackerMaciej", False),
    ("NeuralKinga", False),
    ("EchoFilip", True),
    ("SignalDominika", False),
    ("FrequencyIgor", False),
    ("BipolarKlaudia", True),
    ("SensorRafał", False),
    ("WavePatrycja", False),
    ("SpectrumOskar", True),
    ("NeuroSonia", False),
    ("ApexHubert", False),
    ("BioEwelina", True),
    ("FlashMaks", False),
]

def load_fif_signal(path: str, duration_sec: int = 10, sr: int = 250):
    try:
        import mne
        raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
        picks = [c for c in ['Fp1', 'Fp2', 'O1', 'O2'] if c in raw.ch_names]
        if len(picks) == 4:
            n_samples = sr * duration_sec
            # Skip the first 30 s — electrodes are still settling at the start
            start = min(30 * sr, max(0, raw.n_times - n_samples))
            data = raw.get_data(picks=picks, start=start, stop=start + n_samples).astype(np.float32)
            if data.shape[1] >= sr * 3:
                return data
    except Exception as exc:
        print(f"Warning reading {path}: {exc}")
    return None

def seed_database():
    with app.app_context():
        init_db()
        db = get_db()

        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        fif_files = sorted(glob.glob(os.path.join(data_dir, '*.fif')))
        print(f"[Seed] Znaleziono {len(fif_files)} plików .fif w data/")

        users_created = []

        for idx, (nick, is_solo) in enumerate(DEMO_NICKS):
            # Check if user already exists
            existing = db.execute('SELECT id, stim_signature FROM user WHERE nickname = ?', (nick,)).fetchone()
            if existing and existing['stim_signature']:
                users_created.append((existing['id'], nick, is_solo))
                continue

            if existing:
                # Seeded by an older version (no shared-stimulus signature) —
                # re-record in place, keeping the id and any pair sessions
                uid = existing['id']
            else:
                cur = db.cursor()
                cur.execute('INSERT INTO user (nickname, is_solo) VALUES (?, ?)', (nick, 1 if is_solo else 0))
                uid = cur.lastrowid
                db.commit()

            # Real recording with the same amount of shared stimulation as the
            # app protocol (stimulus markers included); else 10 s / mock fallback
            signal, onsets, labels = None, None, None
            if idx < len(fif_files):
                try:
                    signal, onsets, labels = stim.fif_session(fif_files[idx])
                except Exception as exc:
                    print(f"Warning reading {fif_files[idx]}: {exc}")
                if signal is None:
                    signal = load_fif_signal(fif_files[idx])

            if signal is None:
                # generate distinguished mock
                from eeg import _mock_signal
                signal = _mock_signal(uid)

            _save_embedding(uid, signal, onsets, labels)
            users_created.append((uid, nick, is_solo))
            print(f"  ✓ {'Zaktualizowano' if existing else 'Dodano'} uczestnika: #{uid} {nick} ({'solo' if is_solo else 'para'})")

        # Now create some paired sessions & results for paired users to populate leaderboard
        paired_users = [u for u in users_created if not u[2]]
        print(f"[Seed] Tworzenie par i wyników dla {len(paired_users)} uczestników w trybie par...")

        # Pair them up in twos
        random.seed(42)
        for i in range(0, len(paired_users) - 1, 2):
            u1_id, u1_nick, _ = paired_users[i]
            u2_id, u2_nick, _ = paired_users[i+1]

            # Check if session exists
            sess = db.execute(
                'SELECT id FROM session WHERE (user1_id=? AND user2_id=?) OR (user1_id=? AND user2_id=?)',
                (u1_id, u2_id, u2_id, u1_id)
            ).fetchone()

            if not sess:
                cur = db.cursor()
                cur.execute('INSERT INTO session (user1_id, user2_id) VALUES (?, ?)', (u1_id, u2_id))
                sid = cur.lastrowid

                cur.execute('INSERT INTO result (session_id, score) VALUES (?, 0)', (sid,))
                db.commit()
                print(f"  💖 Para: {u1_nick} & {u2_nick}")

        n_pairs = _refresh_pair_results(db)
        print(f"[Seed] Przeliczono {n_pairs} wyników par (wspólny bodziec)")

        total_users = db.execute('SELECT count(*) FROM user').fetchone()[0]
        total_results = db.execute('SELECT count(*) FROM result').fetchone()[0]
        print(f"\n[Seed] Gotowe! Baza zawiera teraz {total_users} uczestników oraz {total_results} wyników par.")

if __name__ == '__main__':
    seed_database()
