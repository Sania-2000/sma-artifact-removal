import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----- CONFIG -----
INPUT_DIR = "cleaned_chunks"
OUTPUT_BASE_DIR = "noise_chunks"
THRESHOLD_MULTIPLIER = 4
CHANNELS_TO_PLOT = 6

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# ----- PROCESS ALL CLEANED CHUNKS -----
all_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith("_cleaned.csv"))

for filename in all_files:
    print(f"\n>>> Processing: {filename}")
    filepath = os.path.join(INPUT_DIR, filename)
    df = pd.read_csv(filepath)

    chunk_name = filename.replace("_cleaned.csv", "")
    chunk_dir = os.path.join(OUTPUT_BASE_DIR, chunk_name)
    os.makedirs(chunk_dir, exist_ok=True)

    timestamps = df["timestamps"]
    raw_signals = df.drop(columns=["timestamps"])
    noise_only_signals = raw_signals.copy()
    noise_stats = {}

    for col in raw_signals.columns:
        signal = raw_signals[col]
        abs_signal = signal.abs()

        threshold = abs_signal.mean() + THRESHOLD_MULTIPLIER * abs_signal.std()
        spike_mask = abs_signal > threshold

        noise = signal.copy()
        noise[spike_mask] = np.nan
        valid_noise = noise.dropna()

        noise_stats[col] = {
            "max_noise": valid_noise.max() if not valid_noise.empty else 0.0,
            "mean_noise": valid_noise.mean() if not valid_noise.empty else 0.0
        }

        noise_only_signals[col] = noise.fillna(0.0)

    # ----- SAVE NOISE CSV -----
    out_df = noise_only_signals.copy()
    out_df.insert(0, "timestamps", timestamps)
    noise_csv_path = os.path.join(chunk_dir, f"{chunk_name}_noise_only.csv")
    out_df.to_csv(noise_csv_path, index=False, float_format="%.6f")
    print(f"Saved noise-only CSV: {noise_csv_path}")

    # ----- SAVE NOISE STATS -----
    txt_path = os.path.join(chunk_dir, f"{chunk_name}_noise_stats.txt")
    with open(txt_path, "w") as f:
        for ch, stats in noise_stats.items():
            f.write(f"{ch}\tMean Noise: {stats['mean_noise']:.6f}\tMax Noise: {stats['max_noise']:.6f}\n")
    print(f"Saved noise stats: {txt_path}")

    # ----- SAVE PLOT -----
    plot_channels = list(noise_only_signals.columns[:CHANNELS_TO_PLOT])
    fig, axs = plt.subplots(len(plot_channels), 1, figsize=(12, 2.5 * len(plot_channels)), sharex=True)

    for i, col in enumerate(plot_channels):
        axs[i].plot(timestamps, noise_only_signals[col], label=col, color='blue')
        axs[i].set_ylabel("Amplitude")
        axs[i].legend()

    axs[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{filename} - Noise Only (Spikes Removed using abs-based detection)")
    plt.tight_layout()
    plot_path = os.path.join(chunk_dir, f"{chunk_name}_noise_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved noise plot: {plot_path}")
