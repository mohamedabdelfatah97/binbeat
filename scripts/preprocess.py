"""
preprocess.py
Reads raw MIT-BIH records from data/raw/, segments individual heartbeats
into 187-sample windows, applies a frequency threshold to drop rare classes,
and saves X.npy, y.npy, and classes.json to data/processed/.

Usage:
    python3 scripts/preprocess.py
"""

import os
import json
import numpy as np
import wfdb
from collections import Counter

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR      = os.path.join(REPO_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")

# ── config ────────────────────────────────────────────────────────────────────
WINDOW_SIZE       = 187   # samples per heartbeat window
BEFORE_PEAK       = 90    # samples before R-peak
AFTER_PEAK        = 96    # samples after R-peak  (90 + 96 + 1 = 187)
FREQ_THRESHOLD    = 300   # drop classes with fewer than this many samples
EXCLUDE_SYMBOLS = {"+", "~", '"'} # later addition

# standard inter-patient split used in ECG literature
TRAIN_RECORDS = [
    "101", "106", "108", "109", "112", "114", "115", "116",
    "118", "119", "122", "124", "201", "203", "205", "207",
    "208", "209", "215", "220", "223", "230",
]
TEST_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123",
    "200", "202", "210", "212", "213", "214", "219", "221",
    "222", "228", "231", "232", "233", "234",
]

ALL_RECORDS = TRAIN_RECORDS + TEST_RECORDS


def extract_beats(record_name: str, raw_dir: str):
    """Extract all valid beat windows and their labels from one record."""
    record_path = os.path.join(raw_dir, record_name)

    try:
        # load signal — channel 0 (MLII lead)
        record = wfdb.rdrecord(record_path, channels=[0])
        signal = record.p_signal[:, 0]

        # load annotations
        annotation = wfdb.rdann(record_path, "atr")
        r_peaks    = annotation.sample      # sample indices of each beat
        symbols    = annotation.symbol      # annotation symbol for each beat

    except Exception as e:
        print(f"  WARNING: could not read {record_name} — {e}")
        return [], []

    beats, labels = [], []
    n_samples = len(signal)

    for peak, symbol in zip(r_peaks, symbols):
        start = peak - BEFORE_PEAK
        end   = peak + AFTER_PEAK + 1  # +1 so slice gives exactly 187 samples

        # skip beats too close to signal boundaries
        if start < 0 or end > n_samples:
            continue

        beats.append(signal[start:end].astype(np.float32))
        labels.append(symbol)

    return beats, labels


def preprocess(raw_dir: str = RAW_DIR, processed_dir: str = PROCESSED_DIR):
    os.makedirs(processed_dir, exist_ok=True)

    # ── pass 1: collect all beats and count symbol frequencies ────────────────
    print("Pass 1: extracting beats from all records...")
    all_beats  = {"train": [], "test": []}
    all_labels = {"train": [], "test": []}
    symbol_counter = Counter()

    for split, records in [("train", TRAIN_RECORDS), ("test", TEST_RECORDS)]:
        for rec in records:
            beats, labels = extract_beats(rec, raw_dir)
            all_beats[split].extend(beats)
            all_labels[split].extend(labels)
            symbol_counter.update(labels)
            print(f"  {rec} ({split}): {len(beats)} beats")

    # ── apply frequency threshold ──────────────────────────────────────────────
    print(f"\nApplying frequency threshold (min={FREQ_THRESHOLD})...")
    print(f"{'Symbol':<10} {'Count':<10} {'Status'}")
    print("-" * 35)

    valid_symbols = set()
    for symbol, count in sorted(symbol_counter.items(), key=lambda x: -x[1]):
        if symbol in EXCLUDE_SYMBOLS:
            status = "EXCLUDE (not a heartbeat type)"
        elif count < FREQ_THRESHOLD:
            status = "DROP (too rare)"
        else:
            status = "KEEP"
        print(f"  {symbol:<10} {count:<10} {status}")
        if count >= FREQ_THRESHOLD and symbol not in EXCLUDE_SYMBOLS:
            valid_symbols.add(symbol)

    print(f"\nKept {len(valid_symbols)} classes: {sorted(valid_symbols)}")

    # ── build symbol → integer mapping ────────────────────────────────────────
    symbol_to_int = {sym: i for i, sym in enumerate(sorted(valid_symbols))}
    int_to_symbol = {i: sym for sym, i in symbol_to_int.items()}

    # ── pass 2: filter and build final arrays ─────────────────────────────────
    print("\nPass 2: building final arrays...")
    for split in ["train", "test"]:
        X, y = [], []
        for beat, label in zip(all_beats[split], all_labels[split]):
            if label in valid_symbols:
                X.append(beat)
                y.append(symbol_to_int[label])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        np.save(os.path.join(processed_dir, f"X_{split}.npy"), X)
        np.save(os.path.join(processed_dir, f"y_{split}.npy"), y)
        print(f"  {split}: X={X.shape}, y={y.shape}")

    # ── save class mapping ─────────────────────────────────────────────────────
    classes_info = {
        "symbol_to_int": symbol_to_int,
        "int_to_symbol": int_to_symbol,
        "num_classes":   len(valid_symbols),
        "freq_threshold": FREQ_THRESHOLD,
        "dropped_symbols": sorted(
            sym for sym, cnt in symbol_counter.items()
            if cnt < FREQ_THRESHOLD
        ),
    }
    with open(os.path.join(processed_dir, "classes.json"), "w") as f:
        json.dump(classes_info, f, indent=2)

    print(f"\nSaved to {processed_dir}")
    print(f"  X_train.npy, y_train.npy")
    print(f"  X_test.npy,  y_test.npy")
    print(f"  classes.json ({len(valid_symbols)} classes)")
    print("\nPreprocessing complete.")


if __name__ == "__main__":
    preprocess()