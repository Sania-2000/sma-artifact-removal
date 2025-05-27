import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----- CONFIG -----
INPUT_DIR = "CSV chunks"
OUTPUT_DIR = "cleaned_chunks"
PLOTS_DIR = "cleaned_plots"
Z_SCORE_THRESHOLD = 5.5
DURATION = 10 
DOWNSAMPLE_FACTOR = 50
CHANNELS_PER_PLOT = 6
PLOT_START_TIME = 0  

# List of dead channels
DEAD_CHANNELS = [
    "A023", "A024", "A025", "A026", "A027", "A028", "A029", "A030", "A031",
    "B016", "B023", "C023", "C024", "D020", "D023"
]

# ----- SETUP -----
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ----- PROCESS EACH FILE -----
for filename in sorted(os.listdir(INPUT_DIR)):
    if not filename.endswith(".csv"):
        continue

    print(f"\nProcessing {filename}...")
    filepath = os.path.join(INPUT_DIR, filename)
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()  

    timestamps = df["highpass_A-000_timestamps"]

    # Extract all signal columns
    signal_cols = [col for col in df.columns if col.startswith("highpass_") and col.endswith("_values")]

    # Identify dead columns
    dead_cols = [f"highpass_{ch}_values" for ch in DEAD_CHANNELS]
    dead_cols_present = [col for col in signal_cols if col in dead_cols]
    live_cols = [col for col in signal_cols if col not in dead_cols]

    signals = df[signal_cols].copy()

    # Replace dead channels with 0.0
    for col in dead_cols_present:
        signals[col] = 0.0

    # ----- DETECT ARTIFACTS -----
    summed_signal = signals.sum(axis=1)
    mean = summed_signal.mean()
    std = summed_signal.std()
    threshold = mean + Z_SCORE_THRESHOLD * std
    artifact_mask = np.abs(summed_signal) > threshold
    print(f" Detected {artifact_mask.sum()} artifact timepoints.")

    # ----- CLEANING -----
    cleaned_signals = signals.copy()
    for col in live_cols:
        temp = cleaned_signals[col].copy()
        temp[artifact_mask] = np.nan
        cleaned_signals[col] = temp.interpolate(method='linear').bfill().ffill()

    # Drop dead channels entirely from cleaned output
    cleaned_signals.drop(columns=dead_cols_present, inplace=True)

    # Save cleaned file
    cleaned_df = cleaned_signals.copy()
    cleaned_df.insert(0, "timestamps", timestamps)
    output_file = os.path.join(OUTPUT_DIR, filename.replace(".csv", "_cleaned.csv"))
    cleaned_df.fillna(0.0, inplace=True)
    cleaned_df.to_csv(output_file, index=False, float_format="%.6f")
    print(f" Cleaned file saved: {output_file}")

    # ----- PLOTTING -----
    mask_window = (timestamps >= PLOT_START_TIME) & (timestamps <= PLOT_START_TIME + DURATION)
    timestamps_window = timestamps[mask_window][::DOWNSAMPLE_FACTOR]
    cleaned_window = cleaned_signals[mask_window].iloc[::DOWNSAMPLE_FACTOR]

    for i in range(0, len(cleaned_signals.columns), CHANNELS_PER_PLOT):
        group = cleaned_signals.columns[i:i + CHANNELS_PER_PLOT]
        fig, axs = plt.subplots(len(group), 1, figsize=(12, 2.5 * len(group)), sharex=True)
        for j, col in enumerate(group):
            axs[j].plot(timestamps_window, cleaned_window[col], label="Cleaned")
            axs[j].set_title(col)
            axs[j].legend(loc="upper right")
        fig.suptitle(f"{filename} - Cleaned Channels {i+1}–{i+len(group)}")
        plt.tight_layout()
        plot_file = os.path.join(PLOTS_DIR, filename.replace(".csv", f"_CLEANED_channels_{i//CHANNELS_PER_PLOT + 1}.png"))
        plt.savefig(plot_file)
        plt.close()
        print(f"Saved cleaned plot: {plot_file}")
