import os
import re
import pandas as pd

# ---- CONFIG ----
SPIKE_DIR = "spike_results/data"
NOISE_DIR = "noise_from_abs"
OUTPUT_DIR = "snr_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NOISE_THRESHOLD = 1e-3
MAX_VALID_SNR = 1000

# ---- HELPER FUNCTION ----
def extract_value(pattern, line):
    match = re.search(pattern, line)
    return float(match.group(1)) if match else None

# ---- GET ALL CHUNKS ----
chunks = sorted([
    fname.replace("_spikes_abs.txt", "")
    for fname in os.listdir(SPIKE_DIR)
    if fname.endswith("_spikes_abs.txt")
])

# ---- PROCESS EACH CHUNK ----
for CHUNK in chunks:
    print(f"\n🔄 Processing: {CHUNK}")

    SPIKE_FILE = os.path.join(SPIKE_DIR, f"{CHUNK}_spikes_abs.txt")
    NOISE_FILE = os.path.join(NOISE_DIR, f"{CHUNK}_noise_stats.txt")
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{CHUNK}_snr_final.csv")

    if not os.path.exists(SPIKE_FILE) or not os.path.exists(NOISE_FILE):
        print(f"Missing files for {CHUNK}, skipping.")
        continue

    # ---- READ spike data ----
    spike_values = {}
    with open(SPIKE_FILE, "r") as f:
        for line in f:
            ch_match = re.match(r"(\S+)", line)
            if ch_match:
                ch = ch_match.group(1)
                mean_spike = extract_value(r"Mean Amplitude: ([\d\.\-eE]+)", line)
                if mean_spike is not None:
                    spike_values[ch] = mean_spike

    # ---- READ noise data ----
    noise_values = {}
    with open(NOISE_FILE, "r") as f:
        for line in f:
            ch_match = re.match(r"(\S+)", line)
            if ch_match:
                ch = ch_match.group(1)
                mean_noise = extract_value(r"Mean Noise: ([\d\.\-eE]+)", line)
                if mean_noise is not None:
                    noise_values[ch] = mean_noise

    # ---- COMPUTE SNR ----
    snr_results = []
    skipped = []

    for ch in spike_values:
        if ch in noise_values:
            mean_spike = spike_values[ch]
            mean_noise = noise_values[ch]

            if mean_noise < NOISE_THRESHOLD:
                skipped.append((ch, "mean_noise too low"))
                print(f"{CHUNK} | {ch} | Skipped: mean_noise too low")
                continue

            snr = mean_spike / mean_noise

            if snr > MAX_VALID_SNR:
                skipped.append((ch, f"SNR too high: {snr:.2f}"))
                print(f"{CHUNK} | {ch} | Skipped: SNR too high ({snr:.2f})")
                continue

            snr_results.append({
                "chunk": CHUNK,
                "channel": ch,
                "mean_spike": mean_spike,
                "mean_noise": mean_noise,
                "SNR": snr
            })
            print(f"{CHUNK} | {ch} | Mean Spike: {mean_spike:.6f} | Mean Noise: {mean_noise:.6f} | SNR: {snr:.2f}")

    # ---- SAVE ----
    if snr_results:
        pd.DataFrame(snr_results).to_csv(OUTPUT_FILE, index=False)
        print(f"Saved SNRs to: {OUTPUT_FILE}")

    if skipped:
        skipped_file = os.path.join(OUTPUT_DIR, f"{CHUNK}_skipped_channels.csv")
        pd.DataFrame(skipped, columns=["channel", "reason"]).to_csv(skipped_file, index=False)
        print(f"Saved skipped channels to: {skipped_file}")
