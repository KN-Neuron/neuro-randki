import json
import math
import os
import threading

import numpy as np
from flask import (Flask, jsonify, redirect, render_template, request,
                   session as flask_session, url_for)

from database import get_db, init_db
import eeg as eeg_mod
import model as model_mod
from model import cosine_sim, SIMILARITY_THRESHOLD, compute_band_powers
import matching
import shared_stimulus as stim

app = Flask(__name__)
app.config['DATABASE'] = os.path.join(app.root_path, 'neuro_randki.db')
app.secret_key = 'dev_key_for_neuro_randki'

# ── Embedding method config ────────────────────────────────────────────────────
# EMBED_METHOD: 'handcrafted' | 'neural' | 'hybrid'
# NEURAL_CHANNELS: comma-separated channel indices for neural method (e.g. "0,1,2,3")
_EMBED_METHOD   = os.environ.get('EMBED_METHOD', 'handcrafted')
_NEURAL_CH_RAW  = os.environ.get('NEURAL_CHANNELS', '0,1,2,3')
_NEURAL_CHANNELS = [int(c) for c in _NEURAL_CH_RAW.split(',')]
model_mod.EMBED_METHOD = _EMBED_METHOD
print(f'[Config] EMBED_METHOD={_EMBED_METHOD}  NEURAL_CHANNELS={_NEURAL_CHANNELS}')

with app.app_context():
    init_db()

# Pre-load model so first request is fast
threading.Thread(target=model_mod.get_embedding,
                 args=(np.zeros((4, 751), dtype=np.float32),),
                 daemon=True).start()


# ── helpers ───────────────────────────────────────────────────────────────────

def _sim_to_score(sim: float) -> int:
    # Map cosine similarity [-1, 1] → score [0, 100].
    # Uses a soft curve: negative sim → low score, sim≈1 → ~100.
    # Centred so sim=0 → 50, sim=THRESHOLD → ~65 (entry level).
    clamped = max(-1.0, min(1.0, float(sim)))
    # shift & scale to [0,1] then apply mild power curve
    t = (clamped + 1.0) / 2.0          # [0, 1]
    t = t ** 0.7                        # curve: spread out lower values
    return int(round(t * 100))


RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), 'recordings')
os.makedirs(RECORDINGS_DIR, exist_ok=True)


def _signal_for_method(signal: np.ndarray, method: str) -> np.ndarray:
    if method == 'neural':
        ch = [c for c in _NEURAL_CHANNELS if c < signal.shape[0]]
        if len(ch) < 4:
            ch = list(range(min(4, signal.shape[0])))
        return signal[ch, :]
    return signal


def _save_embedding(user_id: int, signal: np.ndarray, onsets=None, labels=None,
                    sr: int = stim.SR):
    method = model_mod.EMBED_METHOD

    # Save raw signal (compressed numpy) + stimulus markers for recompute
    sig_path = os.path.join(RECORDINGS_DIR, f'user_{user_id}.npz')
    if onsets is not None and len(onsets):
        np.savez_compressed(sig_path, signal=signal, onsets=np.asarray(onsets, dtype=np.int64),
                            labels=np.asarray(labels), sr=sr)
    else:
        np.savez_compressed(sig_path, signal=signal)

    db = get_db()
    _store_features(db, user_id, signal, onsets, labels, sr)
    db.execute('UPDATE user SET signal_path=? WHERE id=?', (sig_path, user_id))
    db.commit()


def _store_features(db, user_id: int, signal: np.ndarray, onsets=None, labels=None,
                    sr: int = stim.SR):
    """Embedding, band powers, archetype and shared-stimulus signature."""
    method = model_mod.EMBED_METHOD
    emb   = model_mod.get_embedding(_signal_for_method(signal, method), method=method)
    bands = compute_band_powers(signal, sr)
    arch  = matching.get_archetype(bands)
    sig, quality, n_ep = stim.compute_signature(signal, onsets, labels, sr)
    if onsets is not None and len(onsets):
        print(f'[Stimulus] user={user_id} epochs={n_ep} quality='
              f'{quality if quality is None else round(quality, 2)}')
    db.execute(
        '''UPDATE user SET embedding=?, band_powers=?, archetype=?,
                           stim_signature=?, stim_quality=? WHERE id=?''',
        (json.dumps(emb.tolist()), json.dumps(bands), arch['code'],
         json.dumps(np.round(sig, 4).tolist()) if sig is not None else None,
         quality, user_id)
    )


def _load_recording(path: str):
    """(signal, onsets | None, labels | None, sr) from a saved .npz."""
    with np.load(path, allow_pickle=False) as z:
        if 'onsets' in z.files:
            return z['signal'], z['onsets'], [str(l) for l in z['labels']], int(z['sr'])
        return z['signal'], None, None, stim.SR


_PROFILE_COLS = 'id, nickname, is_solo, archetype, band_powers, embedding, stim_signature, stim_quality'


def _profile(r) -> dict:
    """DB user row → everything matching.compute_affinity needs."""
    bands = json.loads(r['band_powers']) if r['band_powers'] else {}
    return {
        'id': r['id'],
        'nickname': r['nickname'],
        'is_solo': r['is_solo'],
        'archetype': r['archetype'] or matching.get_archetype(bands)['code'],
        'band_powers': bands,
        'embedding': np.array(json.loads(r['embedding']), dtype=np.float32) if r['embedding'] else None,
        'signature': np.array(json.loads(r['stim_signature']), dtype=np.float32) if r['stim_signature'] else None,
        'stim_quality': r['stim_quality'],
    }


def _all_profiles(db) -> list[dict]:
    rows = db.execute(f'SELECT {_PROFILE_COLS} FROM user '
                      'WHERE embedding IS NOT NULL OR band_powers IS NOT NULL').fetchall()
    return [_profile(r) for r in rows]


def _affinity(p1: dict, p2: dict, sig_mean) -> dict:
    return matching.compute_affinity(p1['band_powers'], p1['archetype'],
                                     p2['band_powers'], p2['archetype'],
                                     p1['embedding'], p2['embedding'],
                                     p1['signature'], p2['signature'], sig_mean)


def _load_embedding(user_id: int) -> np.ndarray | None:
    row = get_db().execute('SELECT embedding FROM user WHERE id = ?',
                           (user_id,)).fetchone()
    if row and row['embedding']:
        return np.array(json.loads(row['embedding']), dtype=np.float32)
    return None


def _player_device(player: int) -> str:
    """Return the device name assigned to *player* (1 or 2), or empty string."""
    return flask_session.get(f'device{player}', '') or ''


def _current_device() -> str:
    """Return device for the currently active player / solo user."""
    mode = flask_session.get('mode', 'paired')
    if mode == 'solo':
        return flask_session.get('solo_device', '')
    return _player_device(flask_session.get('current_player', 1))


# ── page routes ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return redirect(url_for('register'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        nick1   = request.form.get('nick1', '').strip() or 'Gracz 1'
        nick2   = request.form.get('nick2', '').strip() or 'Gracz 2'
        device1 = request.form.get('device1', '').strip()
        device2 = request.form.get('device2', '').strip()

        db  = get_db()
        cur = db.cursor()
        cur.execute('INSERT INTO user (nickname) VALUES (?)', (nick1,))
        u1 = cur.lastrowid
        cur.execute('INSERT INTO user (nickname) VALUES (?)', (nick2,))
        u2 = cur.lastrowid
        cur.execute('INSERT INTO session (user1_id, user2_id) VALUES (?, ?)', (u1, u2))
        sid = cur.lastrowid
        db.commit()

        flask_session.update(
            mode='paired', session_id=sid,
            user1_id=u1, user2_id=u2,
            nick1=nick1, nick2=nick2,
            device1=device1, device2=device2,
            current_player=1,
        )
        return redirect(url_for('waiting'))

    return render_template('register.html')


@app.route('/solo', methods=['GET', 'POST'])
def solo():
    if request.method == 'POST':
        nick   = request.form.get('nick', '').strip() or 'Gracz Solo'
        device = request.form.get('device', '').strip()

        db  = get_db()
        cur = db.cursor()
        cur.execute('INSERT INTO user (nickname, is_solo) VALUES (?, 1)', (nick,))
        uid = cur.lastrowid
        db.commit()

        flask_session.update(mode='solo', solo_user_id=uid,
                             solo_nick=nick, solo_device=device)
        return redirect(url_for('waiting'))

    return render_template('solo.html')


@app.route('/waiting')
def waiting():
    mode = flask_session.get('mode', 'paired')

    if mode == 'solo':
        user_id = flask_session.get('solo_user_id')
        if not user_id:
            return redirect(url_for('solo'))
        nick   = flask_session.get('solo_nick', 'Gracz')
        device = flask_session.get('solo_device', '')
        return render_template('waiting.html',
                               player=None, nick=nick, mode='solo',
                               device=device, next_url=url_for('task'))

    player  = flask_session.get('current_player', 1)
    user_id = flask_session.get('user1_id') if player == 1 else flask_session.get('user2_id')
    if not user_id:
        return redirect(url_for('register'))

    nick    = flask_session.get('nick1') if player == 1 else flask_session.get('nick2')
    nick1   = flask_session.get('nick1', 'Gracz 1')
    device  = _player_device(player)
    return render_template('waiting.html',
                           player=player, nick=nick, nick1=nick1,
                           mode='paired', device=device,
                           next_url=url_for('task'))


@app.route('/task')
def task():
    mode = flask_session.get('mode', 'paired')

    if mode == 'solo':
        user_id  = flask_session.get('solo_user_id')
        next_url = url_for('solo_results')
    else:
        player   = flask_session.get('current_player', 1)
        user_id  = (flask_session.get('user1_id') if player == 1
                    else flask_session.get('user2_id'))
        next_url = url_for('waiting') if player == 1 else url_for('results')

    if not user_id:
        return redirect(url_for('register'))

    device = _current_device()
    return render_template('task.html', user_id=user_id,
                           next_url=next_url, device=device,
                           protocol=stim.protocol())


@app.route('/results')
def results():
    sid = flask_session.get('session_id')
    if not sid:
        return redirect(url_for('register'))

    db  = get_db()
    u1, u2 = flask_session.get('user1_id'), flask_session.get('user2_id')

    row1 = db.execute(f'SELECT {_PROFILE_COLS} FROM user WHERE id=?', (u1,)).fetchone() if u1 else None
    row2 = db.execute(f'SELECT {_PROFILE_COLS} FROM user WHERE id=?', (u2,)).fetchone() if u2 else None
    if not row1 or not row2:
        return redirect(url_for('register'))
    p1, p2 = _profile(row1), _profile(row2)
    arch1 = matching.ARCHETYPES.get(p1['archetype'], matching.ARCHETYPES['zen'])
    arch2 = matching.ARCHETYPES.get(p2['archetype'], matching.ARCHETYPES['focus'])

    res = db.execute('SELECT score, similarity, breakdown FROM result WHERE session_id = ?',
                     (sid,)).fetchone()

    if res and res['breakdown']:
        score = res['score']
        breakdown = json.loads(res['breakdown'])
    else:
        sig_mean = stim.population_mean([p['signature'] for p in _all_profiles(db)])
        aff = _affinity(p1, p2, sig_mean)
        score = aff['overall']
        similarity = aff['similarity']
        breakdown = aff

        db.execute('DELETE FROM result WHERE session_id = ?', (sid,))
        db.execute('INSERT INTO result (session_id, score, similarity, breakdown) VALUES (?, ?, ?, ?)',
                   (sid, score, similarity, json.dumps(breakdown)))
        db.commit()

    leaderboard = db.execute(
        '''SELECT u1.nickname AS nick1, u2.nickname AS nick2, r.score, r.breakdown
           FROM result r
           JOIN session s ON r.session_id = s.id
           JOIN user u1 ON s.user1_id = u1.id
           JOIN user u2 ON s.user2_id = u2.id
           ORDER BY r.score DESC LIMIT 10'''
    ).fetchall()

    return render_template('results.html',
                           score=score,
                           nick1=flask_session.get('nick1'),
                           nick2=flask_session.get('nick2'),
                           u1_id=u1,
                           u2_id=u2,
                           arch1=arch1,
                           arch2=arch2,
                           breakdown=breakdown,
                           weak_signal=any(p['signature'] is not None and stim.is_weak(p['stim_quality'])
                                           for p in (p1, p2)),
                           leaderboard=leaderboard,
                           graph_url=url_for('graph'))


@app.route('/solo_results')
def solo_results():
    user_id = flask_session.get('solo_user_id')
    nick    = flask_session.get('solo_nick', 'Gracz')
    db      = get_db()
    user_row = db.execute('SELECT nickname, archetype, band_powers FROM user WHERE id=?', (user_id,)).fetchone() if user_id else None
    arch = None
    if user_row:
        bands = json.loads(user_row['band_powers']) if user_row['band_powers'] else {}
        arch_code = user_row['archetype'] or matching.get_archetype(bands)['code']
        arch = matching.ARCHETYPES.get(arch_code, matching.ARCHETYPES['zen'])

    return render_template('solo_results.html',
                           nick=nick,
                           user_id=user_id,
                           archetype=arch,
                           graph_url=url_for('graph'))


@app.route('/graph')
def graph():
    return render_template('graph.html')


# ── Device API ────────────────────────────────────────────────────────────────

@app.route('/api/devices/scan', methods=['POST'])
def api_devices_scan():
    devices = eeg_mod.scan_devices()
    connected = eeg_mod.list_connected()
    for d in devices:
        d['connected'] = d['name'] in connected
    return jsonify(devices=devices)


@app.route('/api/devices/connect', methods=['POST'])
def api_devices_connect():
    data   = request.get_json(silent=True) or {}
    device = data.get('device', '').strip()
    if not device:
        return jsonify(error='missing device name'), 400
    result = eeg_mod.connect_device(device)
    return jsonify(result)


@app.route('/api/devices/calibrate', methods=['POST'])
def api_devices_calibrate():
    data   = request.get_json(silent=True) or {}
    device = data.get('device', '').strip()
    if not device:
        return jsonify(error='missing device name'), 400
    result = eeg_mod.calibrate_channels(device)
    return jsonify(result)


@app.route('/api/devices/info')
def api_devices_info():
    device = request.args.get('device', '')
    if not device:
        return jsonify(error='missing device'), 400
    return jsonify(eeg_mod.get_device_info(device))


# ── EEG recording API ─────────────────────────────────────────────────────────

@app.route('/api/eeg/connect', methods=['POST'])   # legacy
def api_eeg_connect():
    eeg_mod.connect_async()
    return jsonify(status='connecting')


@app.route('/api/eeg/status')                      # legacy
def api_eeg_status():
    return jsonify(connected=eeg_mod.is_connected())


@app.route('/api/eeg/start', methods=['POST'])
def api_eeg_start():
    data    = request.get_json(silent=True) or {}
    user_id = data.get('user_id')
    device  = data.get('device') or _current_device()
    if not user_id:
        return jsonify(error='missing user_id'), 400
    hw = eeg_mod.start_recording(int(user_id), device)
    return jsonify(status='recording', hardware=hw, device=device)


@app.route('/api/eeg/stop', methods=['POST'])
def api_eeg_stop():
    data    = request.get_json(silent=True) or {}
    user_id = data.get('user_id')
    if not user_id:
        return jsonify(error='missing user_id'), 400

    user_id = int(user_id)

    mode = flask_session.get('mode', 'paired')
    if mode == 'paired' and flask_session.get('current_player', 1) == 1:
        flask_session['current_player'] = 2

    signal, meta = eeg_mod.stop_recording_meta(user_id)
    sr = int(meta['sr'])
    onsets, labels = stim.markers_to_samples(meta['markers'], meta['start'], meta['chunks'], sr)
    if signal is None:
        # No headset data — demo mode: a real data/*.fif participant's
        # recording for as many stimuli as were shown (random noise as last resort)
        sr = stim.SR
        signal, onsets, labels = stim.mock_session(user_id, len(meta['markers']))
        if signal is None:
            signal = eeg_mod._mock_signal(user_id)
    _save_embedding(user_id, signal, onsets, labels, sr)
    n_markers = 0 if onsets is None else len(onsets)
    return jsonify(status='done', user_id=user_id, mock=meta['mock'], markers=n_markers)


@app.route('/api/eeg/mark', methods=['POST'])
def api_eeg_mark():
    """Stimulus onset marker from the task page (one per stimulus)."""
    data    = request.get_json(silent=True) or {}
    user_id = data.get('user_id')
    label   = data.get('label')
    t       = data.get('t')
    if not user_id or label not in stim.COLORS or t is None:
        return jsonify(error='need user_id, label, t'), 400
    ok = eeg_mod.add_marker(int(user_id), label, float(t))
    return jsonify(ok=ok)


# ── Graph data API ────────────────────────────────────────────────────────────

@app.route('/api/graph_data')
def api_graph_data():
    db = get_db()
    nodes_data = _all_profiles(db)

    sessions = db.execute('SELECT user1_id, user2_id FROM session').fetchall()
    session_couples = [{'user1_id': s['user1_id'], 'user2_id': s['user2_id']} for s in sessions]

    graph = matching.build_curated_graph(nodes_data, session_couples)
    return jsonify(graph)


def _refresh_pair_results(db) -> int:
    """Stored pair results cache scores — recompute them for the current population."""
    profiles = {p['id']: p for p in _all_profiles(db)}
    sig_mean = stim.population_mean([p['signature'] for p in profiles.values()])
    pairs = db.execute('SELECT r.id, s.user1_id, s.user2_id FROM result r '
                       'JOIN session s ON r.session_id = s.id').fetchall()
    n = 0
    for row in pairs:
        p1, p2 = profiles.get(row['user1_id']), profiles.get(row['user2_id'])
        if not p1 or not p2:
            continue
        aff = _affinity(p1, p2, sig_mean)
        db.execute('UPDATE result SET score=?, similarity=?, breakdown=? WHERE id=?',
                   (aff['overall'], aff['similarity'], json.dumps(aff), row['id']))
        n += 1
    db.commit()
    return n


@app.route('/admin/recompute', methods=['POST'])
def admin_recompute():
    """Recompute all embeddings from stored raw signals using current method."""
    db      = get_db()
    rows    = db.execute('SELECT id, signal_path FROM user WHERE signal_path IS NOT NULL').fetchall()
    method  = model_mod.EMBED_METHOD
    updated = 0
    for r in rows:
        try:
            sig, onsets, labels, sr = _load_recording(r['signal_path'])
            _store_features(db, r['id'], sig, onsets, labels, sr)
            updated += 1
        except Exception as e:
            print(f'[Recompute] user {r["id"]} failed: {e}')
    db.commit()
    pairs = _refresh_pair_results(db)
    return redirect(url_for('admin', msg=f'Przeliczono {updated} embeddingów metodą [{method}] i {pairs} wyników par'))


@app.route('/api/config/method', methods=['GET', 'POST'])
def api_config_method():
    if request.method == 'POST':
        data   = request.get_json(silent=True) or {}
        method = data.get('method', '').lower()
        if method not in ('neural', 'handcrafted', 'hybrid'):
            return jsonify(error='invalid method'), 400
        model_mod.EMBED_METHOD = method
        if 'channels' in data:
            global _NEURAL_CHANNELS
            _NEURAL_CHANNELS[:] = [int(c) for c in data['channels']]
        print(f'[Config] Method changed to {method}, channels={_NEURAL_CHANNELS}')
        return jsonify(method=method, channels=_NEURAL_CHANNELS)
    return jsonify(method=model_mod.EMBED_METHOD, channels=_NEURAL_CHANNELS)


@app.route('/api/eeg/live')
def api_eeg_live():
    device  = request.args.get('device', '')
    samples = eeg_mod.get_live_samples(device, 250)
    if samples is None:
        return jsonify(data=None)
    # downsample 2× for bandwidth
    ds = [ch[::2] for ch in samples]
    return jsonify(data=ds, sr=125)


@app.route('/api/user_bands/<int:user_id>')
def api_user_bands(user_id):
    row = get_db().execute('SELECT band_powers FROM user WHERE id=?',
                           (user_id,)).fetchone()
    if not row or not row['band_powers']:
        return jsonify(error='no bands'), 404
    return jsonify(json.loads(row['band_powers']))


@app.route('/api/ranking')
def api_ranking():
    db   = get_db()
    rows = db.execute(
        'SELECT id, nickname, is_solo, embedding FROM user WHERE embedding IS NOT NULL'
    ).fetchall()
    embs = {r['id']: np.array(json.loads(r['embedding']), dtype=np.float32) for r in rows}
    counts = {r['id']: {'nickname': r['nickname'], 'is_solo': bool(r['is_solo']),
                         'connections': 0} for r in rows}
    ids = list(embs.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if cosine_sim(embs[ids[i]], embs[ids[j]]) >= SIMILARITY_THRESHOLD:
                counts[ids[i]]['connections'] += 1
                counts[ids[j]]['connections'] += 1
    ranking = sorted(counts.values(), key=lambda x: -x['connections'])
    return jsonify(ranking=ranking)


@app.route('/api/brain_twin/<int:user_id>')
def api_brain_twin(user_id):
    db  = get_db()
    me  = db.execute(f'SELECT {_PROFILE_COLS} FROM user WHERE id=?', (user_id,)).fetchone()
    if not me:
        return jsonify(error='user not found'), 404

    p_me = _profile(me)
    arch_me = matching.ARCHETYPES.get(p_me['archetype'], matching.ARCHETYPES['zen'])
    others = [p for p in _all_profiles(db) if p['id'] != user_id]
    sig_mean = stim.population_mean([p_me['signature']] + [p['signature'] for p in others])

    candidates = []
    for p in others:
        arch_r = matching.ARCHETYPES.get(p['archetype'], matching.ARCHETYPES['zen'])
        aff = _affinity(p_me, p, sig_mean)
        candidates.append({
            'id': p['id'],
            'nickname': p['nickname'],
            'is_solo': bool(p['is_solo']),
            'archetype': arch_r['code'],
            'archetype_name': arch_r['name'],
            'archetype_icon': arch_r['icon'],
            'archetype_color': arch_r['color'],
            'score': aff['overall'],
            'similarity': aff['similarity'],
            'harmony': aff['harmony'],
            'energy': aff['energy'],
            'cognition': aff['cognition'],
            'sync': aff['sync'],
            'bond_type': aff['bond_type'],
            'diagnosis': aff['diagnosis'],
            'strengths': aff['strengths'],
        })

    candidates.sort(key=lambda x: -x['score'])
    best_twin = candidates[0] if candidates else None

    return jsonify({
        'me': {
            'nickname': me['nickname'],
            'archetype': arch_me['code'],
            'archetype_name': arch_me['name'],
            'archetype_icon': arch_me['icon'],
            'headline': arch_me['headline'],
            'desc': arch_me['desc'],
            'has_stim': p_me['signature'] is not None,
            'stim_quality': p_me['stim_quality'],
            'stim_weak': p_me['signature'] is not None and stim.is_weak(p_me['stim_quality']),
        },
        'twin': best_twin,
        'top_candidates': candidates[:3]
    })


@app.route('/admin')
def admin():
    db    = get_db()
    users = db.execute(
        '''SELECT id, nickname, is_solo, archetype,
                  CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_emb,
                  band_powers, stim_quality,
                  CASE WHEN stim_signature IS NULL THEN 0 ELSE 1 END AS has_stim
           FROM user ORDER BY id DESC'''
    ).fetchall()
    return render_template('admin.html', auth=True, users=users,
                           total=len(users),
                           archetypes=matching.ARCHETYPES,
                           stim_quality_min=stim.QUALITY_MIN,
                           stim_protocol=stim.protocol(),
                           embed_method=model_mod.EMBED_METHOD,
                           neural_channels=','.join(str(c) for c in _NEURAL_CHANNELS))


@app.route('/admin/delete/<int:user_id>', methods=['POST'])
def admin_delete(user_id):
    db = get_db()
    db.execute('DELETE FROM user WHERE id=?', (user_id,))
    db.commit()
    return redirect(url_for('admin'))


@app.route('/admin/seed', methods=['POST'])
def admin_seed():
    from seed_data import seed_database
    seed_database()
    return redirect(url_for('admin', msg='Załadowano dane uczestników z bazy EEG!'))


@app.route('/admin/clear', methods=['POST'])
def admin_clear():
    db = get_db()
    db.execute('DELETE FROM result')
    db.execute('DELETE FROM session')
    db.execute('DELETE FROM user')
    db.commit()
    import glob
    for f in glob.glob(os.path.join(RECORDINGS_DIR, 'user_*.npz')):
        try:
            os.remove(f)
        except OSError:
            pass
    return redirect(url_for('admin', msg='Wyczyszczono bazę danych i nagrania.'))


@app.route('/admin/export')
def admin_export():
    import io, csv as csv_mod
    db   = get_db()
    rows = db.execute('SELECT id, nickname, is_solo, archetype, band_powers, stim_quality '
                      'FROM user WHERE embedding IS NOT NULL').fetchall()
    out  = io.StringIO()
    w    = csv_mod.writer(out)
    w.writerow(['id', 'nickname', 'is_solo', 'archetype', 'delta', 'theta', 'alpha', 'beta', 'gamma',
                'stim_quality'])
    for r in rows:
        bands = json.loads(r['band_powers']) if r['band_powers'] else {}
        w.writerow([r['id'], r['nickname'], r['is_solo'], r['archetype'] or '',
                    bands.get('delta',''), bands.get('theta',''),
                    bands.get('alpha',''), bands.get('beta',''), bands.get('gamma',''),
                    '' if r['stim_quality'] is None else round(r['stim_quality'], 3)])
    out.seek(0)
    from flask import Response
    return Response(out.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=neuro_randki_export.csv'})


@app.route('/api/user_viz/<int:user_id>')
def api_user_viz(user_id):
    row = get_db().execute(
        'SELECT embedding FROM user WHERE id = ?', (user_id,)
    ).fetchone()
    if not row or not row['embedding']:
        return jsonify(error='no embedding'), 404
    emb = np.array(json.loads(row['embedding']), dtype=np.float32)
    # Use first 32 values as wave params (freq, amp, phase per wave)
    # Normalise to [0,1] for JS consumption
    emb_min, emb_max = float(emb.min()), float(emb.max())
    span = emb_max - emb_min or 1.0
    params = ((emb[:32] - emb_min) / span).tolist()
    return jsonify(params=params)


if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
