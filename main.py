from muselsl import stream, list_muses, record
from threading import Thread, Event
import time
import os
import asyncio
import atexit
import signal
from merge_session_data import aggregate_session
import subprocess
import sys
from pylsl import resolve_byprop

muses = list_muses()
shutdown_event = Event()


def play_tone():
    """Play a simple notification tone on macOS; fallback to terminal bell."""
    try:
        subprocess.run(['afplay', '/System/Library/Sounds/Ping.aiff'], check=False)
    except Exception:
        try:
            # Terminal bell as last resort
            print('\a', end='', flush=True)
        except Exception:
            pass

def stream_muse():
    """Thread function to stream Muse data (sets its own asyncio loop for Bleak)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        try:
            stream(muses[0]['address'], ppg_enabled=True, acc_enabled=True, gyro_enabled=True)
        except Exception:
            if not shutdown_event.is_set():
                play_tone()
            return
        else:
            # stream() returned (likely disconnected)
            if not shutdown_event.is_set():
                play_tone()
    finally:
        # Intentionally do not close the event loop here to avoid CoreBluetooth callbacks
        # hitting a closed loop during disconnect teardown.
        pass

RECORD_DURATION = 10800


def record_modality(data_source, output_dir, duration=RECORD_DURATION):
    """Record a specific modality (EEG/PPG/ACC/GYRO) to CSV into output_dir for given duration."""
    filename = os.path.join(output_dir, f"{data_source.lower()}.csv")
    record(duration=duration, filename=filename, data_source=data_source)


def _wait_for_lsl_streams(types, timeout_total=30):
    """Return a set of data_source types for which an LSL stream is found within timeout_total seconds."""
    wanted = list(types)
    found = set()
    deadline = time.time() + timeout_total
    while time.time() < deadline and len(found) < len(wanted):
        for typ in wanted:
            if typ in found:
                continue
            streams = resolve_byprop('type', typ, timeout=1)
            if streams:
                print(f"Found {typ} stream.")
                found.add(typ)
        if len(found) < len(wanted):
            time.sleep(1)
    return found

if __name__ == "__main__":
    if not muses:
        raise RuntimeError("No Muse devices found. Ensure the headset is on and discoverable.")

    # Create per-session directory under data/<unix_timestamp>
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_dir, 'data')
    session_ts = str(int(time.time()))
    session_dir = os.path.join(data_root, session_ts)
    os.makedirs(session_dir, exist_ok=True)

    # Start streaming first so the EEG LSL outlet exists before recording resolves it
    stream_thread = Thread(target=stream_muse, daemon=True)
    stream_thread.start()

    # Give the stream a moment to initialize its LSL outlets
    time.sleep(15)

    # Start recording threads only for modalities with present LSL streams
    modalities = ["EEG", "PPG", "ACC", "GYRO"]
    print("Looking for LSL streams before recording...")
    found = _wait_for_lsl_streams(modalities, timeout_total=30)
    if not found:
        print("ERROR: No LSL streams found. Exiting.")
        shutdown_event.set()
        sys.exit(1)

    record_threads = [
        Thread(target=record_modality, args=(source, session_dir, RECORD_DURATION))
        for source in sorted(found)
    ]

    # Ensure aggregation runs on exit or interruption
    _aggregated = {"done": False}

    def _write_aggregate_once():
        if not _aggregated["done"]:
            try:
                aggregate_session(session_dir, output_root=data_root, cleanup=True)
            finally:
                _aggregated["done"] = True

    def _handle_exit(signum=None, frame=None):
        shutdown_event.set()
        _write_aggregate_once()
        raise SystemExit(0)

    atexit.register(_write_aggregate_once)
    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)
    try:
        signal.signal(signal.SIGHUP, _handle_exit)
        signal.signal(signal.SIGQUIT, _handle_exit)
    except Exception:
        pass

    for t in record_threads:
        t.start()

    # Watch the streaming thread; if it ends (disconnect), aggregate without cleanup
    def _watch_stream_and_aggregate():
        stream_thread.join()
        if not shutdown_event.is_set():
            try:
                aggregate_session(session_dir, output_root=data_root, cleanup=False)
            except Exception as e:
                print(f"Warning: Early aggregation failed: {e}")

    watcher_thread = Thread(target=_watch_stream_and_aggregate, daemon=True)
    watcher_thread.start()

    # Listen for 'exit'/'quit' on stdin to force aggregation and terminate
    def _watch_stdin_for_exit():
        try:
            for line in sys.stdin:
                if line.strip().lower() in ('exit', 'quit', 'q'):
                    shutdown_event.set()
                    _write_aggregate_once()
                    os._exit(0)
        except Exception:
            pass

    stdin_thread = Thread(target=_watch_stdin_for_exit, daemon=True)
    stdin_thread.start()

    for t in record_threads:
        t.join()

    # After recordings finish, aggregate the session
    shutdown_event.set()
    _write_aggregate_once()