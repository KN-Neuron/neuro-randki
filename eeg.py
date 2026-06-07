"""BrainAccess EEG — multi-device, automatic channel selection."""
import threading
import numpy as np
import time
from typing import Optional

NUM_MODEL_CHANNELS = 4
DEFAULT_SR = 250
CALIBRATION_SEC = 3  # seconds of signal used for channel ranking

# ── BrainAccess core — init only once ─────────────────────────────────────────
_core_ready = False
_core_lock  = threading.Lock()

def _ensure_core():
    global _core_ready
    with _core_lock:
        if not _core_ready:
            from brainaccess import core
            core.init()
            _core_ready = True

# ── BT scan — one at a time, cached ───────────────────────────────────────────
_scan_lock    = threading.Lock()
_scan_cache: list = []

# ── per-device state ──────────────────────────────────────────────────────────
# key: device name string
# value: {mgr, sr, n_eeg, channels, connected, battery}
_devices: dict = {}
_dev_lock = threading.Lock()

# ── per-recording session ─────────────────────────────────────────────────────
# key: user_id (int)
# value: {data: list[ndarray], active: bool, device: str, channels: list[int]}
_sessions: dict = {}
_sess_lock = threading.Lock()

# ── live preview buffer (written by calibration + recording callbacks) ─────────
# key: device_name → list of (n_eeg, T) chunks, capped at ~3 s
_live: dict = {}
_live_lock = threading.Lock()
_LIVE_MAX_SAMPLES = 750  # 3 s @ 250 Hz


# ─────────────────────────────────────────────────────────────────────────────
#  SCANNING
# ─────────────────────────────────────────────────────────────────────────────

def scan_devices() -> list[dict]:
    """
    Scan Bluetooth for BrainAccess devices.  Only one scan runs at a time;
    concurrent callers get the last cached result immediately.
    Returns [{name: str}].
    """
    global _scan_cache
    if not _scan_lock.acquire(blocking=False):
        # Another scan in progress — return cached result so the caller isn't blocked
        return list(_scan_cache)
    try:
        from brainaccess import core
        _ensure_core()
        found = core.scan()
        _scan_cache = [{'name': d.name} for d in found]
        return list(_scan_cache)
    except Exception as exc:
        print(f'[EEG] scan error: {exc}')
        return list(_scan_cache)
    finally:
        _scan_lock.release()


# ─────────────────────────────────────────────────────────────────────────────
#  CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

def connect_device(device_name: str) -> dict:
    """
    Connect to named device.  Safe to call if already connected.
    Returns: {status, n_channels, sr, battery}
    """
    with _dev_lock:
        dev = _devices.get(device_name)
        if dev and dev.get('connected'):
            return {'status': 'already_connected',
                    'n_channels': dev['n_eeg'], 'sr': dev['sr'],
                    'battery': dev.get('battery', -1)}
    try:
        from brainaccess import core
        from brainaccess.core.eeg_manager import EEGManager
        from brainaccess.core.gain_mode import GainMode
        import brainaccess.core.eeg_channel as eeg_ch

        _ensure_core()
        found = core.scan()
        port = next((d.name for d in found if device_name in d.name), None)
        if port is None:
            return {'status': 'not_found'}

        mgr = EEGManager()
        code = mgr.connect(port)
        if code == 1:
            return {'status': 'connect_failed'}
        if code == 2:
            return {'status': 'firmware_incompatible'}

        feats = mgr.get_device_features()
        n_eeg = feats.electrode_count()
        for i in range(n_eeg):
            mgr.set_channel_enabled(eeg_ch.ELECTRODE_MEASUREMENT + i, True)
            mgr.set_channel_gain(eeg_ch.ELECTRODE_MEASUREMENT + i, GainMode.X8)
        if n_eeg > 0:
            mgr.set_channel_bias(eeg_ch.ELECTRODE_MEASUREMENT + n_eeg - 1, True)
        mgr.set_channel_enabled(eeg_ch.SAMPLE_NUMBER, True)
        mgr.set_channel_enabled(eeg_ch.STREAMING, True)

        sr = mgr.get_sample_frequency()
        mgr.load_config()

        try:
            batt = mgr.get_battery_info().level
        except Exception:
            batt = -1

        with _dev_lock:
            _devices[device_name] = {
                'mgr': mgr,
                'sr': sr,
                'n_eeg': n_eeg,
                'channels': list(range(min(NUM_MODEL_CHANNELS, n_eeg))),
                'connected': True,
                'battery': batt,
            }

        print(f'[EEG] Connected {device_name}: {n_eeg}ch {sr}Hz bat={batt}%')
        return {'status': 'ok', 'n_channels': n_eeg, 'sr': sr, 'battery': batt}

    except Exception as exc:
        print(f'[EEG] connect_device error: {exc}')
        return {'status': 'error', 'message': str(exc)}


def list_connected() -> list[str]:
    with _dev_lock:
        return [k for k, v in _devices.items() if v.get('connected')]


def get_device_info(device_name: str) -> dict:
    with _dev_lock:
        dev = _devices.get(device_name)
        if not dev:
            return {'connected': False}
        return {
            'connected': dev.get('connected', False),
            'n_channels': dev.get('n_eeg', 0),
            'sr': dev.get('sr', DEFAULT_SR),
            'channels': dev.get('channels', []),
            'battery': dev.get('battery', -1),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  CHANNEL CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def _bandpass(data: np.ndarray, sr: int, lo: float = 1.0, hi: float = 40.0) -> np.ndarray:
    """Apply butterworth bandpass.  Silently skips if scipy unavailable."""
    try:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(2, [lo, hi], btype='bandpass', fs=sr, output='sos')
        return sosfiltfilt(sos, data, axis=1)
    except Exception:
        return data


def calibrate_channels(device_name: str,
                       n: int = NUM_MODEL_CHANNELS,
                       duration: float = CALIBRATION_SEC) -> dict:
    """
    Record `duration` seconds on *device_name* and rank channels by signal
    quality.  Updates the device's stored channel selection.

    Returns:
      {channels, variances, n_total}  — or {error} on failure.
    """
    with _dev_lock:
        dev = _devices.get(device_name)
        if not dev or not dev.get('connected'):
            return {'error': 'device not connected'}
        mgr  = dev['mgr']
        n_eeg = dev['n_eeg']
        sr    = dev['sr']

    buf  = []
    lock = threading.Lock()

    def _cb(chunk, chunk_size):
        arr = np.asarray(chunk, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, n_eeg).T
        arr = arr[:n_eeg]
        if arr.size == 0 or arr.shape[1] == 0:
            return
        with lock:
            buf.append(arr.copy())
        with _live_lock:
            _live.setdefault(device_name, []).append(arr.astype(np.float32).copy())
            while sum(a.shape[1] for a in _live[device_name]) > _LIVE_MAX_SAMPLES:
                _live[device_name].pop(0)

    try:
        mgr.set_callback_chunk(_cb)
        mgr.start_stream()
        time.sleep(duration)
        mgr.stop_stream()
        time.sleep(0.15)
    except Exception as exc:
        return {'error': str(exc)}

    with lock:
        if not buf:
            return {'error': 'no data collected'}
        raw = np.concatenate(buf, axis=1).astype(np.float64)   # (n_eeg, T)

    filtered   = _bandpass(raw, sr)
    variances  = np.var(filtered, axis=1)          # one value per channel

    # Quality gate: exclude dead (too flat) and artifact-ridden (too spiky) channels.
    med = float(np.median(variances)) or 1.0
    scores = np.where(
        (variances > med * 0.05) & (variances < med * 25),
        variances, 0.0
    )

    # Pick top-n by score, then sort by index so they form a spatial sequence.
    top_n = int(min(n, np.count_nonzero(scores)))
    if top_n == 0:           # all channels gated out → fall back to first n
        best = list(range(min(n, n_eeg)))
    else:
        best = sorted(np.argsort(scores)[-top_n:].tolist())

    with _dev_lock:
        if device_name in _devices:
            _devices[device_name]['channels'] = best

    print(f'[EEG] {device_name} → best channels: {best}  '
          f'(vars: {[round(variances[c], 2) for c in best]})')
    return {
        'channels':  best,
        'variances': variances.tolist(),
        'n_total':   n_eeg,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  RECORDING
# ─────────────────────────────────────────────────────────────────────────────

def _make_cb(user_id: int, n_eeg: int, device_name: str):
    def _cb(chunk, chunk_size):
        arr = np.asarray(chunk, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, n_eeg).T
        arr = arr[:n_eeg]
        if arr.size == 0 or arr.shape[1] == 0:
            return
        with _sess_lock:
            sess = _sessions.get(user_id)
            if sess and sess['active']:
                sess['data'].append(arr.copy())
        with _live_lock:
            _live.setdefault(device_name, []).append(arr.copy())
            while sum(a.shape[1] for a in _live[device_name]) > _LIVE_MAX_SAMPLES:
                _live[device_name].pop(0)
    return _cb


def start_recording(user_id: int, device_name: str) -> bool:
    """
    Start buffering EEG for *user_id* using *device_name*.
    Channels are taken from the device's last calibration result.
    """
    with _dev_lock:
        dev = _devices.get(device_name, {})
        connected = dev.get('connected', False)
        n_eeg     = dev.get('n_eeg', NUM_MODEL_CHANNELS)
        channels  = dev.get('channels', list(range(NUM_MODEL_CHANNELS)))
        mgr       = dev.get('mgr')

    with _sess_lock:
        _sessions[user_id] = {'data': [], 'active': True,
                              'device': device_name, 'channels': channels}

    if connected and mgr:
        try:
            mgr.set_callback_chunk(_make_cb(user_id, n_eeg, device_name))
            mgr.start_stream()
            print(f'[EEG] Recording: user={user_id} device={device_name} ch={channels}')
            return True
        except Exception as exc:
            print(f'[EEG] start_recording error: {exc}')

    print(f'[EEG] Recording (MOCK): user={user_id}')
    return False


def stop_recording(user_id: int) -> np.ndarray:
    """
    Stop buffering and return (4, T) float32 of the selected EEG channels.
    Falls back to mock data if the session is empty.
    """
    with _sess_lock:
        if user_id in _sessions:
            _sessions[user_id]['active'] = False
            sess_snap = dict(_sessions[user_id])
        else:
            return _mock_signal(user_id)

    device_name = sess_snap.get('device', '')
    channels    = sess_snap.get('channels', list(range(NUM_MODEL_CHANNELS)))

    with _dev_lock:
        dev = _devices.get(device_name, {})
        mgr = dev.get('mgr') if dev.get('connected') else None

    if mgr:
        try:
            mgr.stop_stream()
            time.sleep(0.3)
        except Exception as exc:
            print(f'[EEG] stop_stream error: {exc}')

    with _sess_lock:
        sess = _sessions.pop(user_id, None)

    MIN_SAMPLES = DEFAULT_SR * 5  # at least 5 seconds

    if sess and sess['data']:
        raw = np.concatenate(sess['data'], axis=1).astype(np.float32)  # (n_eeg, T)
        valid = [c for c in channels if c < raw.shape[0]]
        if len(valid) < NUM_MODEL_CHANNELS:
            valid = list(range(min(NUM_MODEL_CHANNELS, raw.shape[0])))
        print(f'[EEG] Stopped: user={user_id} {raw.shape[1]} samples ch={valid}')
        if raw.shape[1] >= MIN_SAMPLES:
            return raw[valid, :]
        print(f'[EEG] Too few samples ({raw.shape[1]}) → mock')

    print(f'[EEG] No data for user {user_id} → mock')
    return _mock_signal(user_id)


def get_live_samples(device_name: str, n: int = 250) -> list | None:
    """Return last n samples per channel as a list-of-lists, or None if no data."""
    with _live_lock:
        chunks = _live.get(device_name, [])
        if not chunks:
            return None
        data = np.concatenate(chunks, axis=1)   # (n_eeg, T)
        data = data[:, -n:]
        # z-score per channel for display
        for i in range(data.shape[0]):
            std = data[i].std() or 1.0
            data[i] = (data[i] - data[i].mean()) / std
        return data.tolist()


def _mock_signal(user_id: int = 0) -> np.ndarray:
    rng = np.random.default_rng(user_id + 42)
    T = int(DEFAULT_SR * 10)
    signal = rng.standard_normal((NUM_MODEL_CHANNELS, T)).astype(np.float32)
    # Shape spectrum so each user has a different dominant band — makes features distinguishable
    from numpy.fft import rfft, irfft, rfftfreq
    freqs = rfftfreq(T, 1.0 / DEFAULT_SR)
    for ch in range(NUM_MODEL_CHANNELS):
        spec = rfft(signal[ch].astype(np.float64))
        dominant = 1.0 + float((user_id * 7 + ch * 3) % 40)
        weights = np.exp(-((freqs - dominant) ** 2) / 25.0) + 0.1
        signal[ch] = irfft(spec * weights, n=T).astype(np.float32)
    return signal


# ─────────────────────────────────────────────────────────────────────────────
#  LEGACY COMPAT  (used by app.py before this refactor)
# ─────────────────────────────────────────────────────────────────────────────

DEVICE_NAME_PRIMARY = 'BA MAXI 012'

def connect_async():
    t = threading.Thread(target=connect_device, args=(DEVICE_NAME_PRIMARY,),
                         daemon=True, name='eeg-connect')
    t.start()

def is_connected() -> bool:
    return DEVICE_NAME_PRIMARY in list_connected()
