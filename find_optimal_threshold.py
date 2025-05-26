#!/usr/bin/env python3
"""
Find optimal threshold for Mode S signal detection
"""

import numpy as np
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt

def analyze_signal_levels():
    """Analyze signal levels to find optimal threshold"""

    # Connect to SDR
    sdr = RtlSdr()
    sdr.sample_rate = 2.0e6
    sdr.center_freq = 1090e6
    sdr.gain = 'auto'

    print("Analyzing signal levels for optimal threshold...")

    try:
        # Collect multiple samples
        all_magnitudes = []

        for i in range(10):
            samples = sdr.read_samples(256*1024)
            magnitude = np.abs(samples)
            all_magnitudes.extend(magnitude)
            print(f"Collected sample {i+1}/10")

        all_magnitudes = np.array(all_magnitudes)

        # Calculate statistics
        mean_mag = np.mean(all_magnitudes)
        std_mag = np.std(all_magnitudes)
        min_mag = np.min(all_magnitudes)
        max_mag = np.max(all_magnitudes)

        # Calculate percentiles
        percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]
        perc_values = np.percentile(all_magnitudes, percentiles)

        print(f"\n📊 SIGNAL STATISTICS:")
        print(f"   Mean: {mean_mag:.6f}")
        print(f"   Std:  {std_mag:.6f}")
        print(f"   Min:  {min_mag:.6f}")
        print(f"   Max:  {max_mag:.6f}")

        print(f"\n📈 PERCENTILES:")
        for p, v in zip(percentiles, perc_values):
            print(f"   {p:5.1f}%: {v:.6f}")

        # Test different thresholds
        print(f"\n🎯 THRESHOLD ANALYSIS:")
        test_thresholds = [
            mean_mag + 1.0 * std_mag,
            mean_mag + 2.0 * std_mag,
            mean_mag + 3.0 * std_mag,
            mean_mag + 4.0 * std_mag,
            mean_mag + 5.0 * std_mag,
            perc_values[4],  # 99th percentile
            perc_values[5],  # 99.5th percentile
            perc_values[6],  # 99.9th percentile
        ]

        threshold_names = [
            "Mean + 1σ",
            "Mean + 2σ",
            "Mean + 3σ",
            "Mean + 4σ",
            "Mean + 5σ",
            "99th percentile",
            "99.5th percentile",
            "99.9th percentile"
        ]

        for name, threshold in zip(threshold_names, test_thresholds):
            above_threshold = np.sum(all_magnitudes > threshold)
            percentage = (above_threshold / len(all_magnitudes)) * 100
            print(f"   {name:18s}: {threshold:.6f} ({percentage:.3f}% above)")

        # Look for signal bursts (potential Mode S)
        print(f"\n🔍 SIGNAL BURST ANALYSIS:")

        # Use a high threshold to find strong signals
        high_threshold = perc_values[5]  # 99.5th percentile

        # Find consecutive samples above threshold (potential pulses)
        above_threshold = all_magnitudes > high_threshold

        # Find start and end of bursts
        burst_starts = []
        burst_lengths = []

        in_burst = False
        burst_start = 0

        for i, is_above in enumerate(above_threshold):
            if is_above and not in_burst:
                # Start of burst
                in_burst = True
                burst_start = i
            elif not is_above and in_burst:
                # End of burst
                in_burst = False
                burst_length = i - burst_start
                burst_starts.append(burst_start)
                burst_lengths.append(burst_length)

        if burst_lengths:
            print(f"   Found {len(burst_lengths)} signal bursts")
            print(f"   Average burst length: {np.mean(burst_lengths):.1f} samples")
            print(f"   Burst length range: {np.min(burst_lengths)} - {np.max(burst_lengths)} samples")

            # Mode S messages should be ~240 samples long at 2 MHz (120 μs * 2 MHz)
            mode_s_length = 240
            tolerance = 50

            mode_s_bursts = [l for l in burst_lengths if abs(l - mode_s_length) < tolerance]
            print(f"   Potential Mode S bursts (≈{mode_s_length}±{tolerance} samples): {len(mode_s_bursts)}")

            if mode_s_bursts:
                print(f"   Mode S burst lengths: {mode_s_bursts[:10]}...")  # Show first 10

        # Recommend optimal threshold
        print(f"\n💡 RECOMMENDATIONS:")

        # For Mode S, we want to catch strong pulses but avoid noise
        # Typically 99th-99.5th percentile works well
        recommended_threshold = perc_values[4]  # 99th percentile
        print(f"   Recommended threshold: {recommended_threshold:.6f} (99th percentile)")
        print(f"   This captures {100 - 99:.1f}% of strongest signals")

        # Alternative: mean + 4σ (catches ~99.99% of normal distribution)
        alt_threshold = mean_mag + 4.0 * std_mag
        print(f"   Alternative threshold: {alt_threshold:.6f} (Mean + 4σ)")

        return recommended_threshold

    finally:
        sdr.close()

if __name__ == "__main__":
    analyze_signal_levels()
