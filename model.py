import os
import numpy as np
import torch
import torch.nn as nn

class MasterEEGNet(nn.Module):
    def __init__(self, num_channels=4, num_classes=22, embedding_dim=128):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 32, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU()
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(32, 64, (num_channels, 1), groups=32, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )
        self.deep = nn.Sequential(
            nn.Conv2d(64, 64, (1, 16), padding=(0, 8), groups=64, bias=False),
            nn.Conv2d(64, 128, (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.deep(x)
        x = self.adaptive_pool(x).view(x.size(0), -1)
        emb = self.fc(x)
        return nn.functional.normalize(emb, p=2, dim=1), None


_model = None

def get_model():
    global _model
    if _model is None:
        _model = MasterEEGNet(num_channels=4, num_classes=22, embedding_dim=128)
        path = os.path.join(os.path.dirname(__file__), 'models', 'openset_master_best.pth')
        if os.path.exists(path):
            _model.load_state_dict(torch.load(path, map_location='cpu'))
            print(f'[Model] Loaded from {path}')
        else:
            print('[Model] WARNING: model file not found, using random weights')
        _model.eval()
    return _model


def get_embedding(signal: np.ndarray) -> np.ndarray:
    """
    signal: (4, T) numpy array — 4 EEG channels
    returns: (128,) L2-normalized embedding
    """
    model = get_model()
    x = signal.copy().astype(np.float32)
    for c in range(x.shape[0]):
        x[c] = (x[c] - x[c].mean()) / (x[c].std() + 1e-8)
    tensor = torch.tensor(x).unsqueeze(0).unsqueeze(0)  # (1, 1, 4, T)
    with torch.no_grad():
        emb, _ = model(tensor)
    return emb.numpy().flatten()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


SIMILARITY_THRESHOLD = 0.3258


def compute_band_powers(signal: np.ndarray, sr: int = 250) -> dict:
    """signal: (4, T) → relative power per EEG band, averaged across channels."""
    from numpy.fft import rfft, rfftfreq
    n = signal.shape[1]
    if n < 2:
        return dict(delta=0.2, theta=0.2, alpha=0.2, beta=0.2, gamma=0.2)
    freqs = rfftfreq(n, d=1.0 / sr)
    psd   = (np.abs(rfft(signal, axis=1)) ** 2).mean(axis=0)

    def band(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(psd[m].mean()) if m.any() else 0.0

    raw   = dict(delta=band(0.5,4), theta=band(4,8),
                 alpha=band(8,13),  beta=band(13,30), gamma=band(30,45))
    total = sum(raw.values()) or 1.0
    return {k: round(v / total, 4) for k, v in raw.items()}
