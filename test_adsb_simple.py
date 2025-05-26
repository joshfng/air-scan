#!/usr/bin/env python3
"""
Simple ADS-B signal test to see what we're actually receiving
"""

import numpy as np
from rtlsdr import RtlSdr
import time

def test_adsb_signals():
    """Simple test to see raw ADS-B signals"""

    # Test different gain settings
    gain_settings = [20.0, 30.0, 40.0, 'auto']

    for gain_setting in gain_settings:
        print(f"\n{'='*60}")
        print(f"TESTING GAIN: {gain_setting}")
        print(f"{'='*60}")

        # Connect to SDR
        sdr = RtlSdr()
        sdr.sample_rate = 2.0e6
        sdr.center_freq = 1090e6

        # Set gain
        if gain_setting == 'auto':
            sdr.gain = 'auto'
            print(f"Using automatic gain control")
        else:
            sdr.gain = gain_setting
            print(f"Using manual gain: {sdr.gain} dB")

        print(f"Sample rate: {sdr.sample_rate/1e6:.1f} MHz")
        print(f"Center frequency: {sdr.center_freq/1e6:.1f} MHz")

        try:
            for i in range(3):  # Just 3 batches per gain setting
                # Get samples
                samples = sdr.read_samples(256*1024)
                magnitude = np.abs(samples)

                # Basic statistics
                mean_mag = np.mean(magnitude)
                max_mag = np.max(magnitude)
                min_mag = np.min(magnitude)
                std_mag = np.std(magnitude)

                # Check for quantization - count unique values
                unique_values = np.unique(magnitude)
                num_unique = len(unique_values)

                print(f"Batch {i+1}: mean={mean_mag:.6f}, max={max_mag:.6f}, "
                      f"min={min_mag:.6f}, std={std_mag:.6f}")
                print(f"         Unique values: {num_unique}")

                if num_unique <= 10:  # If heavily quantized
                    print(f"         Values: {unique_values[:10]}")
                else:
                    print(f"         Sample values: {unique_values[:5]} ... {unique_values[-5:]}")

                # Look for strong signals
                threshold = mean_mag + 2 * std_mag
                strong_signals = np.sum(magnitude > threshold)

                print(f"         Strong signals: {strong_signals}")

                time.sleep(0.2)

        except Exception as e:
            print(f"Error with gain {gain_setting}: {e}")
        finally:
            sdr.close()

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_adsb_signals()
