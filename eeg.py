"""BrainAccess EEG acquisition module with mock fallback."""
import threading
import numpy as np
import time
from typing import Optional

DEVICE_NAME = 'BA MAXI 012'
DEVICE_MAC = 'C8:F0:9E:1C:C3:4A'

# 4 channels selected from 8 available electrodes on BA MAXI 012
# Indices chosen for frontal/central/parietal spread (0-indexed)
EEG_CHANNEL_INDICES = [0, 2, 5, 7]
NUM_MODEL_CHANNELS = 4
DEFAULT_SR = 250  # Hz

# ── global hardware state ─────────────────────────────────────────────────────
_ba_mgr = None
_ba_connected = False
_ba_sr = DEFAULT_SR
_ba_n_eeg = 8
_ba_lock = threading.Lock()
_connect_thread: Optional[threading.Thread] = None

# ── per-recording session buffers ─────────────────────────────────────────────
_sessions: dict = {}   # user_id -> {'data': list[ndarray], 'active': bool}
_sess_lock = threading.Lock()


def _do_connect() -> bool:
    global _ba_mgr, _ba_connected, _ba_sr, _ba_n_eeg
    try:
        from brainaccess import core
        from brainaccess.core.eeg_manager import EEGManager
        from brainaccess.core.gain_mode import GainMode
        import brainaccess.core.eeg_channel as eeg_ch

        core.init()
        devices = core.scan()
        port = next((d.name for d in devices if DEVICE_NAME in d.name), None)
        if port is None:
            print(f'[EEG] Device "{DEVICE_NAME}" not found in scan')
            return False

        mgr = EEGManager()
        status = mgr.connect(port)
        if status == 1:
            print('[EEG] Connection failed')
            return False
        if status == 2:
            print('[EEG] Firmware incompatible')
            return False

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

        with _ba_lock:
            _ba_mgr = mgr
            _ba_sr = sr
            _ba_n_eeg = n_eeg
            _ba_connected = True

        print(f'[EEG] Connected: {port} | {n_eeg} ch | {sr} Hz')
        return True

    except Exception as exc:
        print(f'[EEG] Connection error: {exc}')
        return False


def connect_async():
    """Kick off BT connection in background. Safe to call multiple times."""
    global _connect_thread
    with _ba_lock:
        if _ba_connected:
            return
        if _connect_thread and _connect_thread.is_alive():
            return
    t = threading.Thread(target=_do_connect, daemon=True, name='eeg-connect')
    _connect_thread = t
    t.start()


def is_connected() -> bool:
    with _ba_lock:
        return _ba_connected


def _make_chunk_cb(user_id: int):
    def cb(chunk, chunk_size):
        with _sess_lock:
            sess = _sessions.get(user_id)
            if sess and sess['active']:
                sess['data'].append(chunk[:_ba_n_eeg].copy())
    return cb


def start_recording(user_id: int) -> bool:
    """Start buffering EEG data for *user_id*. Returns True if HW is active."""
    with _sess_lock:
        _sessions[user_id] = {'data': [], 'active': True}

    with _ba_lock:
        connected = _ba_connected
        mgr = _ba_mgr

    if connected and mgr:
        try:
            mgr.set_callback_chunk(_make_chunk_cb(user_id))
            mgr.start_stream()
            print(f'[EEG] Recording started for user {user_id}')
            return True
        except Exception as exc:
            print(f'[EEG] start_recording error: {exc}')

    print(f'[EEG] Recording started (MOCK) for user {user_id}')
    return False


def stop_recording(user_id: int) -> np.ndarray:
    """
    Stop recording and return a (4, T) float32 array of selected EEG channels.
    Falls back to mock data if hardware is unavailable or buffer is empty.
    """
    with _sess_lock:
        if user_id in _sessions:
            _sessions[user_id]['active'] = False

    with _ba_lock:
        connected = _ba_connected
        mgr = _ba_mgr

    if connected and mgr:
        try:
            mgr.stop_stream()
            time.sleep(0.3)
        except Exception as exc:
            print(f'[EEG] stop_stream error: {exc}')

    with _sess_lock:
        sess = _sessions.pop(user_id, None)

    if sess and sess['data']:
        raw = np.hstack(sess['data'])           # (n_eeg, T)
        valid = [i for i in EEG_CHANNEL_INDICES if i < raw.shape[0]]
        if len(valid) < NUM_MODEL_CHANNELS:
            valid = list(range(min(NUM_MODEL_CHANNELS, raw.shape[0])))
        print(f'[EEG] Stopped for user {user_id}: {raw.shape[1]} samples')
        return raw[valid, :].astype(np.float32)

    print(f'[EEG] No data for user {user_id}, returning mock')
    return _mock_signal()


def _mock_signal() -> np.ndarray:
    T = int(DEFAULT_SR * 60)
    return np.random.randn(NUM_MODEL_CHANNELS, T).astype(np.float32)
