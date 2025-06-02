# SMA Artifact Removal

This repository contains a Python script for artifact removal from EEG signal data related to Spinal Muscular Atrophy (SMA). The script detects and mitigates artifacts in EEG recordings stored as CSV files and outputs cleaned versions along with visual plots.

## Features

- Detects artifacts using z-score thresholding on the summed signal
- Replaces affected signal values with interpolated data
- Generates visual plots of cleaned EEG channels for review

## Configuration

You can modify the following parameters in `main.py` to adjust behavior:

- `Z_SCORE_THRESHOLD` — sets the sensitivity for artifact detection
- `DURATION` — duration of the time window for plotting (in seconds)
- `DOWNSAMPLE_FACTOR` — controls resolution of plot data
- `CHANNELS_PER_PLOT` — sets how many channels are displayed per plot