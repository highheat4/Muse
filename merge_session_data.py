import os
import sys
import argparse
import numpy as np
import pandas as pd
import shutil
from typing import Optional, Dict, List

# ---------- Helpers ----------

def _spread_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Spread identical timestamps evenly up to the next unique base timestamp (for ACC/GYRO)."""
    if 'timestamps' not in df.columns or df.empty:
        return df
    ts = df['timestamps'].astype(float).to_numpy()
    base = pd.Series(ts).drop_duplicates().to_numpy()
    diffs = np.diff(base)
    diffs = diffs[diffs > 0]
    fallback_gap = float(np.median(diffs)) if diffs.size else 0.01  # 10 ms fallback

    new_ts = ts.copy()
    i = 0
    n_total = ts.size
    while i < n_total:
        t0 = ts[i]
        j = i + 1
        while j < n_total and ts[j] == t0:
            j += 1
        n_run = j - i
        t1 = ts[j] if j < n_total else (t0 + fallback_gap)
        gap = t1 - t0 if (t1 - t0) > 0 else fallback_gap
        if n_run > 1:
            step = gap / float(max(n_run, 1))
            for k in range(1, n_run):
                new_ts[i + k] = t0 + step * k
        i = j
    out = df.copy()
    out['timestamps'] = new_ts
    return out

def _load_and_bucket(csv_path: str, prefix: str, decimals: int = 3, spread_dupes: bool = False) -> Optional[pd.DataFrame]:
    """Load a modality CSV, bucket timestamps to desired decimals, prefix channel names, and return grouped means."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if 'timestamps' not in df.columns or df.empty:
        return None
    df['timestamps'] = df['timestamps'].astype(float)
    if spread_dupes:
        df = _spread_duplicate_timestamps(df)
    df['bucket'] = np.round(df['timestamps'], decimals)

    # Numeric channel columns (exclude timestamps & bucket)
    numeric_cols = [c for c in df.columns if c not in ('timestamps', 'bucket')]
    rename_map: Dict[str, str] = {}
    for c in numeric_cols:
        clean = c.replace(' ', '')
        rename_map[c] = f"{prefix}_{clean}"
    df = df.rename(columns=rename_map).drop(columns=['timestamps'])

    grouped = df.groupby('bucket', as_index=True).mean(numeric_only=True)
    grouped.index.name = 'bucket'
    return grouped

def _align(df: Optional[pd.DataFrame], target_index: pd.Index) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(index=target_index)
    out = df.reindex(target_index)
    out = out.interpolate(method='index').ffill().bfill()
    return out

# ---------- Aggregator ----------

def aggregate_session(session_dir: str, output_root: Optional[str] = None, cleanup: bool = True) -> Optional[str]:
    """
    Aggregate Muse2 session into ONE Parquet with columns:
      - eeg_raw_* (from eeg.csv)
      - eeg_clean_* (from eeg_cleaned.csv)  [preferred timeline]
      - ppg_*, acc_*, gyro_*
    Output: {output_root}/{session_ts}.parquet
    Deletes the session folder if cleanup=True.
    """
    session_dir = os.path.abspath(session_dir)
    if not os.path.isdir(session_dir):
        print(f"Session directory not found: {session_dir}")
        return None

    session_name = os.path.basename(session_dir.rstrip(os.sep))
    data_root = output_root or os.path.dirname(session_dir)
    os.makedirs(data_root, exist_ok=True)
    parquet_path = os.path.join(data_root, f"{session_name}.parquet")

    # Paths
    eeg_raw_path   = os.path.join(session_dir, 'eeg.csv')
    eeg_clean_path = os.path.join(session_dir, 'eeg_cleaned.csv')  # must be created by main.py
    ppg_path  = os.path.join(session_dir, 'ppg.csv')
    acc_path  = os.path.join(session_dir, 'acc.csv')
    gyro_path = os.path.join(session_dir, 'gyro.csv')

    # Load EEG (both)
    eeg_clean = _load_and_bucket(eeg_clean_path, 'eeg_clean', decimals=3, spread_dupes=False) if os.path.exists(eeg_clean_path) else None
    eeg_raw   = _load_and_bucket(eeg_raw_path,   'eeg_raw',   decimals=3, spread_dupes=False) if os.path.exists(eeg_raw_path)   else None

    # Define target index: prefer cleaned EEG timeline, else raw EEG
    if eeg_clean is not None and not eeg_clean.empty:
        target_index = eeg_clean.index
    elif eeg_raw is not None and not eeg_raw.empty:
        target_index = eeg_raw.index
    else:
        print("No EEG found; cannot define timeline.")
        return None

    # Load other modalities
    ppg  = _load_and_bucket(ppg_path,  'ppg',  decimals=3, spread_dupes=False)
    acc  = _load_and_bucket(acc_path,  'acc',  decimals=3, spread_dupes=True)   # spread duplicate coarse ts
    gyro = _load_and_bucket(gyro_path, 'gyro', decimals=3, spread_dupes=True)

    # Align and join
    combined = pd.DataFrame(index=target_index)
    for part in (eeg_raw, eeg_clean, ppg, acc, gyro):
        combined = combined.join(_align(part, target_index), how='left')

    combined = combined.sort_index().reset_index().rename(columns={'bucket': 'timestamp'})
    combined['timestamp'] = combined['timestamp'].astype(float).round(3)

    # Write Parquet (require pyarrow)
    try:
        # Quantize all numeric columns to 3 decimals EXCEPT cleaned EEG columns
        eeg_clean_cols = [c for c in combined.columns if c.startswith('eeg_clean_')]
        numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_round = [c for c in numeric_cols if c not in eeg_clean_cols]
        if cols_to_round:
            combined[cols_to_round] = combined[cols_to_round].round(3)
        combined.to_parquet(parquet_path, index=False)  # engine='pyarrow' by default if installed
    except Exception as e:
        print("Failed to write Parquet. Please install pyarrow: pip install pyarrow")
        raise

    print(f"Wrote aggregated Parquet: {parquet_path}")

    if cleanup:
        try:
            shutil.rmtree(session_dir, ignore_errors=True)
            print(f"Cleaned up session directory: {session_dir}")
        except Exception as e:
            # With ignore_errors=True, this should rarely trigger; keep for completeness
            print(f"Warning: Could not clean up session directory {session_dir}: {e}")

    return parquet_path

# ---------- CLI ----------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate a Muse2 session directory into a single Parquet file.")
    p.add_argument("session", help="Path to session dir (e.g., data/1762365268) or just the timestamp.")
    p.add_argument("--data-root", default=None, help="Root data dir; default is parent of the session folder.")
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if os.path.isdir(args.session):
        session_dir = args.session
        data_root = args.data_root or os.path.dirname(os.path.abspath(session_dir))
    else:
        data_root = args.data_root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        session_dir = os.path.join(data_root, args.session)
    result = aggregate_session(session_dir, output_root=data_root, cleanup=True)
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main())
