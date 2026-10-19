"""
Shared-stimulus protocol — everyone watches the same colour sequence and we
compare *how their brains respond to it* (event-locked inter-subject
similarity), instead of comparing resting spectra.

Validated on the 30 recordings in data/: with ~6 min of stimuli the similarity
graph reproduces between two halves of a session (Spearman rho ≈ 0.6), and it
is not driven by headset fit (rho ≈ 0.02 with amplitude / mains-noise graph).
Resting spectral similarity from a short recording reproduced at rho ≈ 0.1.
"""
import glob
import os
from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfiltfilt

SR = 250
COLORS = {
    'gray':   '#808080',
    'red':    '#ff0000',
    'green':  '#00b000',
    'blue':   '#0050ff',
    'yellow': '#ffe600',
}
CONDS = list(COLORS)
STIM_SEC = 3.0                    # each stimulus stays on screen 3 s (as in data/)
N_STIMULI = int(os.environ.get('STIM_TRIALS', 120))   # 120 × 3 s = 6 min
EPOCH = (-0.2, 1.0)               # seconds around onset
SIG_SR = 50                       # signature resolution (Hz)
# Fixed hardware delay (Bluetooth transport + display) that timing alone
# cannot reveal. Identical for everyone on the same headset, so comparisons
# between live participants are unaffected; set it so live recordings line up
# with the data/ demo population (measure once, e.g. with a photodiode).
LATENCY_MS = float(os.environ.get('EEG_LATENCY_MS', 0))
MIN_EPOCHS_PER_COND = 3
# Split-half r below this → flagged as weak signal in the UI. Not used to
# exclude people: on data/ dropping weak signatures did not improve test-retest
# stability (rho 0.62 all vs 0.57 trusted-only).
QUALITY_MIN = 0.10

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FIF_CHANNELS = ['Fp1', 'Fp2', 'O1', 'O2']


# ── protocol ──────────────────────────────────────────────────────────────────

def build_sequence(n: int = N_STIMULI, seed: int = 2024) -> list[str]:
    """Gray alternating with a colour, colours balanced and shuffled with a
    fixed seed — identical for every participant."""
    rng = np.random.default_rng(seed)
    n_col = max(4, n // 2)
    colours = (['red', 'green', 'blue', 'yellow'] * (n_col // 4 + 1))[:n_col]
    rng.shuffle(colours)
    seq = []
    for c in colours:
        seq += ['gray', c]
    return seq[:max(n, 8)]


def protocol() -> dict:
    seq = build_sequence()
    return {'sequence': seq, 'colors': COLORS, 'stim_ms': int(STIM_SEC * 1000),
            'total_sec': round(len(seq) * STIM_SEC)}


# ── timing: browser markers → sample indices ─────────────────────────────────

def markers_to_samples(markers: list, start: float, chunk_log: list,
                       sr: int = SR) -> tuple[np.ndarray, list]:
    """
    markers:   [{'label', 't_browser' (ms, performance.now), 't_server' (s, monotonic)}]
    start:     server monotonic time when recording started
    chunk_log: [(arrival monotonic time, cumulative samples after chunk)]

    Browser timestamps are precise (taken right after the frame is painted) but
    on another clock; server receipt times share the EEG clock but carry
    network jitter. Offset = min(server - browser) ≈ clock offset + minimal
    latency, so onset_server = t_browser + offset keeps the browser precision.
    """
    if not markers:
        return np.array([], dtype=int), []
    tb = np.array([m['t_browser'] for m in markers], dtype=np.float64) / 1000.0
    ts = np.array([m['t_server'] for m in markers], dtype=np.float64)
    onsets = tb + float(np.min(ts - tb))

    if len(chunk_log) >= 10:
        t, n = np.array(chunk_log, dtype=np.float64).T
        a, b = np.polyfit(t, n, 1)               # samples ≈ a·time + b
        # Chunks only ever arrive *late* (Bluetooth lag + jitter), so the
        # least-delayed chunks — the upper envelope of samples-vs-time — mark
        # the true acquisition time; a plain fit would sit at the mean lag.
        b += float(np.percentile(n - (a * t + b), 95))
        idx = a * onsets + b + LATENCY_MS / 1000.0 * sr
    else:
        idx = (onsets - start) * sr
    return np.round(idx).astype(int), [m['label'] for m in markers]


# ── signature ─────────────────────────────────────────────────────────────────

def _epochs(signal: np.ndarray, onsets: np.ndarray, labels: list, sr: int):
    x = np.asarray(signal, dtype=np.float64)
    x = sosfiltfilt(butter(4, [1, 20], btype='bandpass', fs=sr, output='sos'), x, axis=1)
    a, b = int(EPOCH[0] * sr), int(EPOCH[1] * sr)
    eps, labs = [], []
    for s, lab in zip(onsets, labels):
        if lab not in COLORS or s + a < 0 or s + b > x.shape[1]:
            continue
        e = x[:, s + a:s + b]
        eps.append(e - e[:, :-a].mean(axis=1, keepdims=True))   # baseline
        labs.append(lab)
    if not eps:
        return None, None
    eps = np.array(eps)
    # Unit-agnostic artifact rejection: drop epochs far above the typical peak
    peaks = np.abs(eps).max(axis=(1, 2))
    keep = peaks < 6 * np.median(peaks)
    return eps[keep], np.array(labs)[keep]


def _average(eps, labs, step) -> np.ndarray | None:
    parts = []
    for c in CONDS:
        m = labs == c
        if m.sum() < MIN_EPOCHS_PER_COND:
            return None
        parts.append(eps[m].mean(axis=0)[:, ::step].ravel())
    return np.concatenate(parts)


def compute_signature(signal: np.ndarray, onsets, labels, sr: int = SR):
    """
    Trial-averaged response to each colour, all channels, at 50 Hz.
    Returns (signature | None, quality | None, n_epochs) where quality is the
    split-half (odd vs even trials) correlation of the signature.
    """
    if signal is None or onsets is None or len(onsets) == 0:
        return None, None, 0
    eps, labs = _epochs(signal, np.asarray(onsets, dtype=int), list(labels), sr)
    if eps is None or labs is None:
        return None, None, 0
    step = max(1, sr // SIG_SR)
    sig = _average(eps, labs, step)
    if sig is None:
        return None, None, len(eps)
    # Split-half within each colour (the sequence alternates gray/colour, so a
    # global odd/even split would put every gray trial in one half)
    halves = [np.concatenate([np.flatnonzero(labs == c)[h::2] for c in CONDS]) for h in (0, 1)]
    odd, even = (_average(eps[h], labs[h], step) for h in halves)
    quality = float(np.corrcoef(odd, even)[0, 1]) if odd is not None and even is not None else None
    return sig.astype(np.float32), quality, len(eps)


def is_weak(quality) -> bool:
    return quality is None or quality < QUALITY_MIN


def population_mean(signatures: list) -> np.ndarray | None:
    """Response everyone shares — subtracted so similarity reflects the
    individual part of the response (only same-shape signatures are used)."""
    sigs = [np.asarray(s) for s in signatures if s is not None]
    if len(sigs) < 3:
        return None
    lengths = [len(s) for s in sigs]
    common = max(set(lengths), key=lengths.count)
    return np.mean([s for s in sigs if len(s) == common], axis=0)


def signature_similarity(s1, s2, mean=None) -> float | None:
    if s1 is None or s2 is None or len(s1) != len(s2):
        return None
    a, b = np.asarray(s1, dtype=np.float64), np.asarray(s2, dtype=np.float64)
    if mean is not None and len(mean) == len(a):
        a, b = a - mean, b - mean
    a, b = a - a.mean(), b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else None


# Calibrated on data/ with the app protocol (120 stimuli): pairwise similarity
# of mean-removed signatures has p5 ≈ -0.33, median ≈ -0.03, p95 ≈ +0.35.
# Test-retest between two independent 6-min blocks: graph rho 0.62, same
# closest match 33% (chance 3%), top-3 overlap 28% (chance 10%).
SYNC_LO, SYNC_HI = -0.45, 0.45


def sync_score(r: float) -> int:
    """Signature similarity → 'Wspólna reakcja' score (40..99)."""
    t = (r - SYNC_LO) / (SYNC_HI - SYNC_LO)
    return int(round(40 + 59 * min(1.0, max(0.0, t))))


# ── demo / mock mode: realistic responses from data/*.fif ────────────────────

@lru_cache(maxsize=8)
def _fif(path: str):
    import mne
    mne.set_log_level('ERROR')
    raw = mne.io.read_raw_fif(path, preload=True)
    x = np.asarray(raw.get_data(picks=FIF_CHANNELS), dtype=np.float32)
    sr = int(raw.info['sfreq'])
    ev = [(int(round(o * sr)), d) for o, d in zip(raw.annotations.onset, raw.annotations.description)
          if d in COLORS]
    return x, sr, ev


def fif_files() -> list[str]:
    return sorted(glob.glob(os.path.join(DATA_DIR, '*.fif')))


def mock_session(user_id: int, n_stimuli: int = N_STIMULI):
    """
    Demo mode without a headset: a contiguous chunk of one data/ participant's
    real recording with *their own* stimulus markers, cut to as many stimuli
    as the app session showed. Returns (signal, onsets, labels) or Nones.

    Uses the block *after* the one the seeded demo user got, so a mock
    participant is a re-measurement of that person, not an exact duplicate.

    (Splicing real responses under the app's timeline was tried and rejected:
    the splice points are locked to the onsets and create a fake, perfectly
    repeatable 'response'.)
    """
    files = fif_files()
    if not files or n_stimuli < 4 * MIN_EPOCHS_PER_COND:
        return None, None, None
    try:
        x, on, lab = fif_session(files[user_id % len(files)], n_stimuli, skip_stimuli=N_STIMULI)
        if x is None or len(on) < n_stimuli // 2:      # recording too short → first block
            x, on, lab = fif_session(files[user_id % len(files)], n_stimuli)
        return x, on, lab
    except Exception as exc:
        print(f'[Stimulus] mock from fif failed: {exc}')
        return None, None, None


def fif_session(path: str, n_stimuli: int = N_STIMULI, skip_sec: float = 30.0,
                skip_stimuli: int = 0):
    """Cut a participant's recording to *n_stimuli* stimuli (after *skip_sec*
    and *skip_stimuli*) — the same amount of stimulation the app gives.
    Returns (signal, onsets_in_samples, labels)."""
    x, sr, ev = _fif(path)
    ev = [(s, d) for s, d in ev if s >= skip_sec * sr][skip_stimuli:skip_stimuli + n_stimuli]
    if not ev:
        return None, None, None
    start = max(0, ev[0][0] - 2 * sr)
    stop = min(x.shape[1], ev[-1][0] + int(STIM_SEC * sr))
    onsets = np.array([s - start for s, _ in ev], dtype=int)
    return x[:, start:stop], onsets, [d for _, d in ev]
