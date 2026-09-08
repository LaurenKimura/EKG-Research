"""
dataset.py
----------
Loads and preprocesses EKG heartbeat data for classification.

Two modes:
1. load_mitbih_real()  -> Downloads and parses the real MIT-BIH Arrhythmia
   Database from PhysioNet using the `wfdb` package. Requires internet
   access to physionet.org.
2. generate_synthetic_ecg() -> Generates synthetic ECG-like heartbeats
   (normal vs. abnormal) for testing the pipeline when PhysioNet is not
   reachable (e.g. in a sandboxed environment).

Both return the same shape of data so the rest of the pipeline (model,
training, evaluation) does not need to change depending on data source.
"""

import numpy as np

BEAT_LEN = 180  # samples per heartbeat window (standard for MIT-BIH-based work)

# MIT-BIH annotation symbols grouped into 5 standard AAMI classes
AAMI_CLASSES = {
    'N': 0,  # Normal beat (N, L, R, e, j)
    'S': 1,  # Supraventricular ectopic beat (A, a, J, S)
    'V': 2,  # Ventricular ectopic beat (V, E)
    'F': 3,  # Fusion beat (F)
    'Q': 4,  # Unknown / paced (Q, /, f)
}
SYMBOL_TO_CLASS = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    'Q': 'Q', '/': 'Q', 'f': 'Q',
}
CLASS_NAMES = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown/Paced']

# Standard MIT-BIH record numbers (48 total recordings)
MITBIH_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234',
]


def load_mitbih_real(records=None, pn_dir='mitdb'):
    """
    Downloads real MIT-BIH data via wfdb and segments it into labeled
    heartbeat windows. Requires internet access to physionet.org.

    Returns:
        X: np.ndarray, shape (n_beats, BEAT_LEN)
        y: np.ndarray, shape (n_beats,) integer class labels 0-4
    """
    import wfdb

    if records is None:
        records = MITBIH_RECORDS

    X, y = [], []
    for rec_name in records:
        try:
            record = wfdb.rdrecord(rec_name, pn_dir=pn_dir)
            annotation = wfdb.rdann(rec_name, 'atr', pn_dir=pn_dir)
        except Exception as e:
            print(f"  skipping record {rec_name}: {e}")
            continue

        signal = record.p_signal[:, 0]  # use first channel (usually MLII lead)
        half = BEAT_LEN // 2

        for sample_idx, symbol in zip(annotation.sample, annotation.symbol):
            cls = SYMBOL_TO_CLASS.get(symbol)
            if cls is None:
                continue  # skip non-beat annotations (e.g. rhythm change markers)
            start, end = sample_idx - half, sample_idx + half
            if start < 0 or end > len(signal):
                continue
            beat = signal[start:end]
            X.append(beat)
            y.append(AAMI_CLASSES[cls])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def generate_synthetic_ecg(n_samples=4000, seed=42):
    """
    Generates synthetic ECG-like heartbeat windows for pipeline testing
    when the real MIT-BIH dataset is unreachable. NOT real patient data —
    only used to verify the code runs correctly end-to-end.

    Normal beats: clean, regular QRS-like pulse shape.
    Abnormal beats: distorted shape, extra pulses, or shifted timing,
    loosely mimicking premature/ventricular beat morphology.

    Returns:
        X: np.ndarray, shape (n_samples, BEAT_LEN)
        y: np.ndarray, shape (n_samples,) integer class labels 0-4
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(-1, 1, BEAT_LEN)

    def qrs_pulse(center=0.0, width=0.08, amp=1.0):
        return amp * np.exp(-((t - center) ** 2) / (2 * width ** 2))

    X, y = [], []
    class_probs = [0.60, 0.12, 0.18, 0.05, 0.05]  # roughly mimics MIT-BIH imbalance
    for _ in range(n_samples):
        cls = rng.choice(5, p=class_probs)
        noise = rng.normal(0, 0.03, BEAT_LEN)

        if cls == 0:  # Normal
            beat = qrs_pulse(0.0, 0.06, 1.0) + 0.15 * qrs_pulse(-0.3, 0.15, 0.3)
        elif cls == 1:  # Supraventricular - slightly early, smaller P-wave shape
            shift = rng.uniform(-0.15, -0.05)
            beat = qrs_pulse(shift, 0.06, 0.9) + 0.25 * qrs_pulse(shift - 0.25, 0.1, 0.4)
        elif cls == 2:  # Ventricular - wide, tall, bizarre morphology
            beat = qrs_pulse(0.0, 0.16, 1.4)
        elif cls == 3:  # Fusion - mix of normal + ventricular shape
            beat = 0.5 * qrs_pulse(0.0, 0.06, 1.0) + 0.5 * qrs_pulse(0.05, 0.16, 1.2)
        else:  # Unknown/paced - sharp spike + flat segment
            beat = qrs_pulse(0.0, 0.02, 1.6)

        beat = beat + noise
        X.append(beat)
        y.append(cls)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def normalize(X):
    """Per-beat z-score normalization."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mean) / std