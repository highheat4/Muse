import os
import sys
import time
import csv
import json
import argparse
import asyncio
import atexit
import signal
from threading import Thread, Event, Lock
import numpy as np
from typing import List, Optional, Tuple

# Acquisition (single BLE connection)
from muselsl import stream, list_muses, record
from pylsl import resolve_byprop

# Offline EEG cleaning (mandatory)
try:
    from brainflow.board_shim import BoardShim
    from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
    HAS_BRAINFLOW = True
except Exception:
    BoardShim = None  # type: ignore
    DataFilter = None  # type: ignore
    FilterTypes = None  # type: ignore
    DetrendOperations = None  # type: ignore
    HAS_BRAINFLOW = False

# Aggregator (writes Parquet and deletes session dir)
from merge_session_data import aggregate_session as run_aggregator

# ---- Defaults (Muse 2 BLE = board_id 38; US mains=60 Hz) ----
MUSE2_BOARD_ID = 38
MAINS_HZ = 60
LSL_WAIT_SEC = 45

stop_event = Event()

# ------------------------
# Helpers
# ------------------------

def _prepare_dirs(data_root: Optional[str]) -> Tuple[str, str]:
    root = data_root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    ts = str(int(time.time()))
    session_dir = os.path.join(root, ts)
    os.makedirs(session_dir, exist_ok=True)
    return root, session_dir


def _discover_first_muse() -> Optional[str]:
    muses = list_muses()
    if not muses:
        return None
    return muses[0].get("address") or muses[0].get("mac_address") or muses[0].get("device_address")


def _stream_thread(address: str) -> None:
    """Run MuseLSL stream with all modalities enabled; returns when disconnected."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        stream(address, ppg_enabled=True, acc_enabled=True, gyro_enabled=True)
    except Exception as e:
        print(f"Muse stream ended: {e}")


def _wait_for_lsl(types: List[str], timeout_sec: int) -> List[str]:
    want = list(types)
    found: List[str] = []
    deadline = time.time() + timeout_sec
    while time.time() < deadline and len(found) < len(want):
        for t in want:
            if t in found:
                continue
            if resolve_byprop('type', t, timeout=1):
                print(f"Found {t} stream.")
                found.append(t)
        if len(found) < len(want):
            time.sleep(1)
    return found


def _record_one(src: str, session_dir: str, duration: int) -> None:
    filename = os.path.join(session_dir, f"{src.lower()}.csv")
    record(duration=duration, filename=filename, data_source=src)


def _brainflow_clean_eeg_keep_headers(session_dir: str, board_id: int = MUSE2_BOARD_ID, mains_hz: int = MAINS_HZ) -> str:
    """BrainFlow 5.19.0-only cleaner: detrend → notch (center/bandwidth) → bandpass (low/high).
    - perform_bandstop(data, sampling_rate, center_freq, band_width, order, filter_type, ripple)
    - perform_bandpass(data, sampling_rate, low_freq, high_freq, order, filter_type, ripple)
    """
    if not HAS_BRAINFLOW:
        raise RuntimeError("BrainFlow is required for offline EEG cleaning. Install: pip install brainflow")

    eeg_csv = os.path.join(session_dir, "eeg.csv")
    if not os.path.exists(eeg_csv):
        raise FileNotFoundError("eeg.csv not found for cleaning")

    timestamps: List[float] = []
    chans: List[List[float]] = []
    chan_headers: List[str] = []
    with open(eeg_csv, "r", newline="") as fh:
        r = csv.reader(fh)
        header = next(r, None)
        if not header or len(header) < 2:
            raise RuntimeError("Invalid eeg.csv header")
        chan_headers = header[1:]
        n_ch = len(chan_headers)
        chans = [[] for _ in range(n_ch)]
        for row in r:
            try:
                timestamps.append(float(row[0]))
                for i in range(n_ch):
                    chans[i].append(float(row[i + 1]))
            except Exception:
                continue

    # If no data, write an empty cleaned file with headers and return gracefully
    if len(timestamps) == 0 or all(len(ch) == 0 for ch in chans):
        out_csv = os.path.join(session_dir, "eeg_cleaned.csv")
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["timestamps"] + chan_headers)
        print(f"No EEG data to clean; wrote empty cleaned EEG -> {out_csv}")
        return out_csv

    try:
        sfreq = BoardShim.get_sampling_rate(board_id)
    except Exception:
        sfreq = 256  # Muse2 default fallback

    notch_center = 60.0 if mains_hz not in (50, 60) else float(mains_hz)
    notch_bw = 2.0  # Hz (±1 Hz around mains)
    order = 4       # BrainFlow allows 1..8

    # Filter wrappers using only the start/stop (bandstop) and low/high (bandpass) variants
    def _apply_bandstop(buf, fs, center_hz: float, bandwidth_hz: float, order: int) -> None:
        start = float(center_hz) - float(bandwidth_hz) / 2.0
        stop = float(center_hz) + float(bandwidth_hz) / 2.0
        if stop <= start:
            stop = start + 1.0
        DataFilter.perform_bandstop(buf, fs, start, stop, int(order), FilterTypes.BUTTERWORTH.value, 0)

    # Bandpass using low/high only
    def _apply_bandpass(buf, fs, low_hz: float, high_hz: float, order: int) -> None:
        lo = float(low_hz)
        hi = float(high_hz)
        if hi <= lo:
            hi = lo + 1.0
        DataFilter.perform_bandpass(buf, fs, lo, hi, int(order), FilterTypes.BUTTERWORTH.value, 0)

    def _clean(sig: List[float]) -> List[float]:
        buf = np.asarray(sig, dtype=np.float64)
        # Detrend
        DataFilter.detrend(buf, DetrendOperations.CONSTANT.value)
        # Notch @ mains (support both API variants)
        _apply_bandstop(buf, sfreq, notch_center, notch_bw, order)
        # Bandpass 1–40 Hz (support both API variants)
        _apply_bandpass(buf, sfreq, 1.0, 40.0, order)
        return buf.tolist()

    cleaned = [_clean(ch) for ch in chans]

    out_csv = os.path.join(session_dir, "eeg_cleaned.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamps"] + chan_headers)
        for i in range(len(timestamps)):
            w.writerow([timestamps[i]] + [cleaned[j][i] for j in range(len(cleaned))])

    meta = {
        "brainflow_version": "5.19.0",
        "sampling_rate": sfreq,
        "notch_center_hz": notch_center,
        "notch_bw_hz": notch_bw,
        "bandpass_hz": [1.0, 40.0],
        "filter_order": order,
        "filter_type": "BUTTERWORTH",
        "channels": chan_headers,
        "n_samples": len(timestamps),
    }
    with open(os.path.join(session_dir, "brainflow_summary.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"BrainFlow (5.19.0): wrote cleaned EEG -> {out_csv}")
    return out_csv
 


def _finalize_session(session_dir: str, data_root: str) -> Optional[str]:
    """Clean EEG, aggregate to Parquet named after the session folder, delete session dir."""
    try:
        _brainflow_clean_eeg_keep_headers(session_dir)
    except Exception as e:
        print(f"EEG cleaning failed: {e}")
        return None
    try:
        result = run_aggregator(session_dir, output_root=data_root, cleanup=True)
        return result
    except Exception as e:
        print(f"Aggregation failed: {e}")
        return None


# ------------------------
# CLI
# ------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Muse2 capture → EEG clean → aggregate to single Parquet; reconnect on drop; 'q'/'quit'/'exit' to stop.")
    p.add_argument("--duration", type=int, default=10800, help="Per-session recording duration in seconds (default 1800).")
    p.add_argument("--data-root", default=None, help="Root dir for data/ (default: ./data)")
    return p.parse_args(argv)


# ------------------------
# Main loop with reconnect + stdin commands
# ------------------------

def main(argv=None) -> int:
    args = parse_args(argv)
    data_root = args.data_root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    state = {"session_dir": None, "stream_thread": None, "rec_threads": []}
    finalize_lock = Lock()
    finalized_event = Event()

    def finalize_once():
        with finalize_lock:
            if finalized_event.is_set():
                return None
            finalized_event.set()
        if state["session_dir"]:
            return _finalize_session(state["session_dir"], data_root)
        return None

    def _stdin_watcher():
        try:
            for line in sys.stdin:
                cmd = (line or "").strip().lower()
                if cmd in ("q", "quit", "exit"):
                    print("Command received: shutting down after finalizing current session...")
                    stop_event.set()
                    # Join recorders briefly, then finalize
                    for t in list(state["rec_threads"]):
                        try:
                            t.join(timeout=2)
                        except Exception:
                            pass
                    finalize_once()
                    os._exit(0)
        except Exception:
            pass

    Thread(target=_stdin_watcher, daemon=True).start()

    def _handle_signal(signum=None, frame=None):
        print("Signal received: finalizing and exiting...")
        stop_event.set()
        for t in list(state["rec_threads"]):
            try:
                t.join(timeout=2)
            except Exception:
                pass
        finalize_once()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    # Reconnect loop: start a session, if stream drops, finalize and start a new one,
    # until user types q/quit/exit or sends SIGINT/SIGTERM
    wanted = ["EEG", "PPG", "ACC", "GYRO"]

    while not stop_event.is_set():
        address = _discover_first_muse()
        if not address:
            print("No Muse found. Retrying in 5s...")
            time.sleep(5)
            continue

        # Create session dir
        _, session_dir = _prepare_dirs(data_root)
        state["session_dir"] = session_dir
        state["rec_threads"] = []
        # reset finalization gate for this session
        finalized_event = Event()

        with open(os.path.join(session_dir, "session_meta.json"), "w") as fh:
            json.dump({"address": address, "start_unix": int(time.time()), "duration_sec": int(args.duration), "board_id": MUSE2_BOARD_ID}, fh, indent=2)

        # Start streaming
        st = Thread(target=_stream_thread, args=(address,), daemon=True)
        st.start()
        state["stream_thread"] = st

        # Allow outlets to come up, then check
        time.sleep(8)
        print("Looking for LSL streams (EEG, PPG, ACC, GYRO)...")
        found = _wait_for_lsl(wanted, timeout_sec=LSL_WAIT_SEC)
        missing = [w for w in wanted if w not in found]
        if missing:
            print(f"Missing streams {missing}. Finalizing empty session and retrying...")
            # Nothing to record; finalize (will likely fail cleaning if eeg.csv doesn't exist)
            try:
                finalize_once()
            except Exception:
                pass
            # Wait a bit and try again
            time.sleep(5)
            continue

        # Start per-modality recorders (they will end early if stream dies)
        for src in wanted:
            t = Thread(target=_record_one, args=(src, session_dir, int(args.duration)))
            t.start()
            state["rec_threads"].append(t)

        # Watch for natural end: either recorders finish (duration met) or stream thread ends (disconnect)
        # We'll poll the stream thread; if it exits before duration, it's a disconnect.
        start_ts = time.time()
        while True:
            # If user asked to stop, break and finalize
            if stop_event.is_set():
                break
            # If stream ended (disconnect), break
            if not st.is_alive():
                print("Stream disconnected; finalizing session...")
                break
            # If all recorders finished, break
            if all(not t.is_alive() for t in state["rec_threads"]):
                print("Recording threads finished; finalizing session...")
                break
            time.sleep(0.5)

        # Join recorders briefly
        for t in state["rec_threads"]:
            try:
                t.join(timeout=2)
            except Exception:
                pass

        # Finalize current session (clean + aggregate + delete)
        result = finalize_once()
        if result is not None:
            print(f"Session finalized: {result}")
        else:
            print("Session finalization already handled or failed.")

        # If we're stopping, exit loop; else, reconnect and start a new session
        if stop_event.is_set():
            break
        print("Reconnecting in 3s...")
        time.sleep(3)

    print("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
