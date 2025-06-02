import pandas as pd
import numpy as np
import os
import re

# ---- CONFIG ----
INPUT_DIR = "spike_results/data"
OUTPUT_BASE_DIR = "noise_from_abs"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# ---- DETECT CHUNKS ----
all_chunks = sorted([
    fname.replace("_abs.csv", "")
    for fname in os.listdir(INPUT_DIR)
    if fname.endswith("_abs.csv")
])

# ---- PROCESS EACH CHUNK ----
for CHUNK in all_chunks:
    print(f"\nProcessing: {CHUNK}")
    
    ABS_CSV = os.path.join(INPUT_DIR, f"{CHUNK}_abs.csv")
    SPIKE_FILE = os.path.join(INPUT_DIR, f"{CHUNK}_spikes_abs.txt")
    OUTPUT_NOISE_TXT = os.path.join(OUTPUT_BASE_DIR, f"{CHUNK}_noise_stats.txt")

    # Skip if files missing
    if not os.path.exists(ABS_CSV) or not os.path.exists(SPIKE_FILE):
        print(f"Missing files for {CHUNK}, skipping.")
        continue

    # ---- LOAD SIGNAL ----
    df = pd.read_csv(ABS_CSV)
    timestamps = df["timestamps"]
    signal_data = df.drop(columns=["timestamps"])

    # ---- PARSE SPIKES ----
    spike_indices = {}
    with open(SPIKE_FILE, "r") as f:
        for line in f:
            ch_match = re.match(r"(\S+)", line)
            if ch_match:
                ch = ch_match.group(1)
                indices = re.findall(r"\d+", line.split("spike_indices")[-1])
                spike_indices[ch] = [int(i) for i in indices]

    # ---- MASK SPIKES AND COMPUTE NOISE ----
    noise_stats = {}
    for ch in signal_data.columns:
        signal = signal_data[ch].copy()

        if ch not in spike_indices:
            print(f"Channel {ch} not found in spike file, skipping.")
            continue

        spike_idx = spike_indices[ch]
        noise = signal.copy()
        noise.iloc[spike_idx] = np.nan

        valid_noise = noise.dropna()
        mean_abs_noise = valid_noise.abs().mean()
        max_abs_noise = valid_noise.abs().max()

        noise_stats[ch] = {
            "mean_noise": mean_abs_noise,
            "max_noise": max_abs_noise
        }

    # ---- SAVE TO FILE ----
    with open(OUTPUT_NOISE_TXT, "w") as f:
        for ch, stats in noise_stats.items():
            f.write(f"{ch}\tMean Noise: {stats['mean_noise']:.6f}\tMax Noise: {stats['max_noise']:.6f}\n")

    print(f"Saved noise stats to {OUTPUT_NOISE_TXT}")
