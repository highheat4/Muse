# Muse2_ToolKit
Tools to capture, clean, aggregate, and visualize data from a Muse 2 headset, with a focus on enabling personal analysis of focus and meditative states. museLSL alone doesn't offer the end-to-end workflow I want, and BrainFlow's Muse support often requires extra hardware. This toolkit bridges that gap by:
- Storing raw multi-stream data (EEG, PPG, ACC, GYRO) and cleaned EEG in a single, compact Parquet file per session
- Providing a reconnect-resilient recorder and offline EEG cleaning (notch + bandpass) using BrainFlow
- Offering a muselsl-style visualization CLI that lets you explore any time window, toggle channels, and interactively resize the view
- Laying the groundwork for future analytics on focus/meditation metrics

 # Usage
 
 ## Setup
 - Create/activate a Python 3.11 environment.
 - Install dependencies:
   - pip install -r requirements.txt
   - If BrainFlow isn’t installed by the requirements file on your system, install it separately: pip install brainflow
 
 ## Record a session
 - Start recording with reconnect + graceful shutdown:
   - python main.py --duration 1800
   - Default output root is ./data; each session goes into its own timestamped folder.
 - Stop recording:
   - Type q (or quit/exit) in the terminal, or send Ctrl+C (SIGINT).
 - What happens on finalize:
   - EEG is cleaned offline (notch + bandpass) using BrainFlow.
   - All modalities are aggregated into one Parquet file named after the session folder.
   - The session folder is removed after successful aggregation (Parquet remains in data/).
 
 ## Visualize a session
 - Muselsl-style stacked view with interactive controls:
   - python visualize_parquet.py /path/to/session.parquet
   - Default window is last 4 minutes; absolute timestamps on x-axis.
   - Use the time slider at the bottom to resize the window.
   - Use group checkboxes to toggle individual channels; layout auto-fills hidden rows.
 - Useful flags:
   - Select a custom window:
     - --start 10m --duration 6m
     - or --start 120 --end 480 (seconds)
   - Prefer raw EEG or hide groups:
     - --prefer-raw, --no-clean, --no-raw, --no-ppg, --no-gyro, --no-acc
   - Disable interactivity (static plot):
     - --static
   - Save to an image instead of showing:
     - --output plot.png
   - Pick specific columns (overrides groups):
     - --columns eeg_clean_TP9,eeg_clean_AF7


# Commits

## V0.2
Used brainflow to add cleaned eeg columns. Data stored in parquet files now for better compression. Added basic visualization (python visualize_parquet.py /path/to/parquet).

## V0.1
Noticed granularity differences in streamed data; EEG data is the most granular, so interpolated data from the other three files and combined csv readings into one output. Linear interpolation for ACC, GYRO, and PPG data. 

Also added handling for Muse disconnect to store the data easily.

## V0
Currently, just stores the data streamed from my Muse 2 to the data folder.

