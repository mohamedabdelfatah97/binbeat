"""
download_data.py
Downloads all 48 MIT-BIH Arrhythmia Database records from PhysioNet
into data/raw/ using the wfdb library.

Usage:
    python3 scripts/download_data.py
"""

import os
import wfdb

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR   = os.path.join(REPO_ROOT, "data", "raw")

# ── all 48 MIT-BIH record names ───────────────────────────────────────────────
RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]

def download_all(raw_dir: str = RAW_DIR) -> None:
    os.makedirs(raw_dir, exist_ok=True)
    print(f"Downloading {len(RECORDS)} MIT-BIH records to {raw_dir}\n")

    for i, rec in enumerate(RECORDS, 1):
        target = os.path.join(raw_dir, rec)
        # skip if all 3 files already exist
        if all(os.path.exists(f"{target}.{ext}") for ext in ("dat",)):
            print(f"[{i:02d}/{len(RECORDS)}] {rec} — already exists, skipping")
            continue

        try:
            wfdb.dl_database(
                "mitdb",
                dl_dir=raw_dir,
                records=[rec],
                annotators=["atr"],
            )
            print(f"[{i:02d}/{len(RECORDS)}] {rec} — done")
        except Exception as e:
            print(f"[{i:02d}/{len(RECORDS)}] {rec} — FAILED: {e}")

    print("\nDownload complete.")
    print(f"Files saved to: {raw_dir}")


if __name__ == "__main__":
    download_all()