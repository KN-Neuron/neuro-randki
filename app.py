import json
import math
import os
import threading

import numpy as np
from flask import (Flask, jsonify, redirect, render_template, request,
                   session as flask_session, url_for)

from database import get_db, init_db
import eeg as eeg_mod
from model import get_embedding, cosine_sim, SIMILARITY_THRESHOLD

app = Flask(__name__)
app.config['DATABASE'] = os.path.join(app.root_path, 'neuro_randki.db')
app.secret_key = 'dev_key_for_neuro_randki'

with app.app_context():
    init_db()

# Pre-load model so first request is fast
threading.Thread(target=get_embedding,
                 args=(np.zeros((4, 751), dtype=np.float32),),
                 daemon=True).start()


# ── helpers ───────────────────────────────────────────────────────────────────

def _sim_to_score(sim: float) -> int:
    return int(100 / (1 + math.exp(-5 * sim)))


def _save_embedding(user_id: int, signal: np.ndarray):
    emb = get_embedding(signal)
    db = get_db()
    db.execute('UPDATE user SET embedding = ? WHERE id = ?',
               (json.dumps(emb.tolist()), user_id))
    db.commit()
    return emb


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
        nick   = flask_session.get('solo_nick', 'Gracz')
        device = flask_session.get('solo_device', '')
        return render_template('waiting.html',
                               player=None, nick=nick, mode='solo',
                               device=device, next_url=url_for('task'))

    player  = flask_session.get('current_player', 1)
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

    device = _current_device()
    return render_template('task.html', user_id=user_id,
                           next_url=next_url, device=device)


@app.route('/results')
def results():
    sid = flask_session.get('session_id')
    if not sid:
        return redirect(url_for('register'))

    db  = get_db()
    res = db.execute('SELECT score, similarity FROM result WHERE session_id = ?',
                     (sid,)).fetchone()
    if res:
        score, similarity = res['score'], res['similarity']
    else:
        u1, u2 = flask_session.get('user1_id'), flask_session.get('user2_id')
        emb1, emb2 = _load_embedding(u1), _load_embedding(u2)

        if emb1 is not None and emb2 is not None:
            similarity = cosine_sim(emb1, emb2)
            score      = _sim_to_score(similarity)
        else:
            import random
            score, similarity = random.randint(55, 99), None

        db.execute('INSERT INTO result (session_id, score, similarity) VALUES (?, ?, ?)',
                   (sid, score, similarity))
        db.commit()

    leaderboard = db.execute(
        '''SELECT u1.nickname AS nick1, u2.nickname AS nick2, r.score
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
                           leaderboard=leaderboard,
                           graph_url=url_for('graph'))


@app.route('/solo_results')
def solo_results():
    return render_template('solo_results.html',
                           nick=flask_session.get('solo_nick', 'Gracz'),
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

    signal = eeg_mod.stop_recording(user_id)
    _save_embedding(user_id, signal)
    return jsonify(status='done', user_id=user_id)


# ── Graph data API ────────────────────────────────────────────────────────────

@app.route('/api/graph_data')
def api_graph_data():
    db   = get_db()
    rows = db.execute(
        'SELECT id, nickname, embedding, is_solo FROM user WHERE embedding IS NOT NULL'
    ).fetchall()

    nodes, embeddings = [], []
    for r in rows:
        nodes.append({'id': r['id'], 'nickname': r['nickname'],
                      'is_solo': bool(r['is_solo'])})
        embeddings.append(np.array(json.loads(r['embedding']), dtype=np.float32))

    links = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            sim = cosine_sim(embeddings[i], embeddings[j])
            if sim >= SIMILARITY_THRESHOLD:
                links.append({
                    'source': nodes[i]['id'],
                    'target': nodes[j]['id'],
                    'similarity': round(sim, 4),
                    'score':      _sim_to_score(sim),
                })

    return jsonify(nodes=nodes, links=links, threshold=SIMILARITY_THRESHOLD)


if __name__ == '__main__':
    app.run(debug=True)
