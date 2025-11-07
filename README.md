# Muse

# Commits

## V0.2
Used brainflow to add cleaned eeg columns. Data stored in parquet files now for better compression. Added basic visualization (python visualize_parquet.py /path/to/parquet).

## V0.1
Noticed granularity differences in streamed data; EEG data is the most granular, so interpolated data from the other three files and combined csv readings into one output. Linear interpolation for ACC, GYRO, and PPG data. 

Also added handling for Muse disconnect to store the data easily.

## V0
Currently, just stores the data streamed from my Muse 2 to the data folder.

