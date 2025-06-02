import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----- CONFIG -----
INPUT_DIR = "cleaned_chunks"
OUTPUT_DATA_DIR = "spike_results/data"
OUTPUT_PLOTS_DIR = "spike_results/plots"
THRESHOLD_MULTIPLIER = 4
CHANNELS_TO_PLOT = 6

os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

# ----- PROCESS EACH FILE -----
for filename in sorted(os.listdir(INPUT_DIR)):
    if not filename.endswith("_cleaned.csv"):
        continue

    print(f"\nProcessing: {filename}")
    filepath = os.path.join(INPUT_DIR, filename)
    df = pd.read_csv(filepath)

    timestamps = df["timestamps"]
    raw_signals = df.drop(columns=["timestamps"])
    abs_signals = raw_signals.abs()

    stats_raw = {}
    stats_abs = {}

    for col in raw_signals.columns:
        raw = raw_signals[col]
        abs_signal = abs_signals[col]

        # --- Raw signal spike detection ---
        threshold_raw = raw.mean() + THRESHOLD_MULTIPLIER * raw.std()
        spike_mask_raw = raw > threshold_raw
        spikes_raw = raw[spike_mask_raw]

        stats_raw[col] = {
            "spike_count": spike_mask_raw.sum(),
            "max_amplitude": spikes_raw.max() if not spikes_raw.empty else 0.0,
            "mean_spike": spikes_raw.mean() if not spikes_raw.empty else 0.0,
            "spike_indices": spikes_raw.index.tolist()
        }

        # --- Abs signal spike detection ---
        # threshold_abs = abs_signal.mean() + THRESHOLD_MULTIPLIER * abs_signal.std()
        # spike_mask_abs = abs_signal > threshold_abs
        # spikes_abs = abs_signal[spike_mask_abs]

        # stats_abs[col] = {
        #     "spike_count": spike_mask_abs.sum(),
        #     "max_amplitude": spikes_abs.max() if not spikes_abs.empty else 0.0,
        #     "mean_spike": spikes_abs.mean() if not spikes_abs.empty else 0.0,
        #     "spike_indices": spikes_abs.index.tolist()
        # }

    # ----- SAVE RAW SIGNAL CSV -----
    raw_output = raw_signals.copy()
    raw_output.insert(0, "timestamps", timestamps)
    raw_csv_path = os.path.join(OUTPUT_DATA_DIR, filename.replace("_cleaned.csv", "_raw.csv"))
    raw_output.to_csv(raw_csv_path, index=False, float_format="%.6f")
    print(f" Saved raw signal CSV: {raw_csv_path}")

    # # ----- SAVE ABSOLUTE SIGNAL CSV -----
    # abs_output = abs_signals.copy()
    # abs_output.insert(0, "timestamps", timestamps)
    # abs_csv_path = os.path.join(OUTPUT_DATA_DIR, filename.replace("_cleaned.csv", "_abs.csv"))
    # abs_output.to_csv(abs_csv_path, index=False, float_format="%.6f")
    # print(f" Saved absolute signal CSV: {abs_csv_path}")

    # ----- SAVE SPIKE STATS TXT (RAW) -----
    txt_raw = os.path.join(OUTPUT_DATA_DIR, filename.replace("_cleaned.csv", "_spikes_raw.txt"))
    with open(txt_raw, "w") as f:
        for ch, data in stats_raw.items():
            f.write(f"{ch}\tSpikes: {data['spike_count']}\t"
                    f"Max Amplitude: {data['max_amplitude']:.3f}\t"
                    f"Mean Amplitude: {data['mean_spike']:.3f}\n")
    print(f" Saved spike stats (raw): {txt_raw}")

    # # ----- SAVE SPIKE STATS TXT (ABS) -----
    # txt_abs = os.path.join(OUTPUT_DATA_DIR, filename.replace("_cleaned.csv", "_spikes_abs.txt"))
    # with open(txt_abs, "w") as f:
    #     for ch, data in stats_abs.items():
    #         f.write(f"{ch}\tSpikes: {data['spike_count']}\t"
    #                 f"Max Amplitude: {data['max_amplitude']:.3f}\t"
    #                 f"Mean Amplitude: {data['mean_spike']:.3f}\n")
    # print(f" Saved spike stats (abs): {txt_abs}")

    # ----- PLOT ABS SIGNAL -----
    plot_channels = list(raw_signals.columns[:CHANNELS_TO_PLOT])
    fig, axs = plt.subplots(len(plot_channels), 1, figsize=(12, 2.5 * len(plot_channels)), sharex=True)

    for i, col in enumerate(plot_channels):
        axs[i].plot(timestamps, abs_signals[col], label=col)
        axs[i].scatter(timestamps[stats_abs[col]["spike_indices"]],
                       abs_signals[col].iloc[stats_abs[col]["spike_indices"]],
                       color='red', s=10, label="Spikes (abs)")
        axs[i].legend()
        axs[i].set_ylabel("Amplitude")

    axs[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{filename} - Spikes")
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_PLOTS_DIR, filename.replace(".csv", "_spikes_abs.png"))
    plt.savefig(plot_path)
    plt.close()
    print(f" Saved plot: {plot_path}")
