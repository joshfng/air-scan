#!/usr/bin/env python3
"""
ADS-B Signal Analyzer
Comprehensive tool to diagnose ADS-B reception issues
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from rtlsdr import RtlSdr
from scipy import signal
from scipy.fft import fft, fftfreq


class ADSBSignalAnalyzer:
    def __init__(self, sample_rate=2.0e6):
        self.sample_rate = sample_rate
        self.center_freq = 1090e6
        self.sdr = None

    def connect_sdr(self, device_index=0):
        """Connect to RTL-SDR device"""
        try:
            self.sdr = RtlSdr(device_index)
            self.sdr.sample_rate = self.sample_rate
            self.sdr.center_freq = self.center_freq
            print(f"✅ Connected to SDR device {device_index}")
            print(f"   Tuner: {self.sdr.get_tuner_type()}")
            print(f"   Sample rate: {self.sample_rate/1e6:.1f} MHz")
            print(f"   Center frequency: {self.center_freq/1e6:.1f} MHz")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to SDR device: {e}")
            return False

    def test_gain_levels(self):
        """Test different gain levels to find optimal setting"""
        print("\n🔧 TESTING GAIN LEVELS")
        print("=" * 50)

        # Get available gains
        gains = self.sdr.get_gains()
        print(f"Available gains: {[g/10.0 for g in gains]} dB")

        results = []

        for gain in gains[::2]:  # Test every other gain to save time
            self.sdr.gain = gain
            time.sleep(0.1)  # Allow settling

            # Collect samples
            samples = self.sdr.read_samples(256 * 1024)
            magnitude = np.abs(samples)

            # Calculate statistics
            mean_power = np.mean(magnitude)
            max_power = np.max(magnitude)
            std_power = np.std(magnitude)
            dynamic_range = max_power / mean_power if mean_power > 0 else 0

            results.append(
                {
                    "gain": gain / 10.0,
                    "mean": mean_power,
                    "max": max_power,
                    "std": std_power,
                    "dynamic_range": dynamic_range,
                }
            )

            print(
                f"Gain {gain/10.0:4.1f} dB: mean={mean_power:.4f}, max={max_power:.4f}, "
                f"std={std_power:.4f}, DR={dynamic_range:.1f}"
            )

        # Find optimal gain (good dynamic range without saturation)
        best_gain = max(
            results, key=lambda x: x["dynamic_range"] if x["max"] < 0.9 else 0
        )
        print(f"\n💡 Recommended gain: {best_gain['gain']:.1f} dB")

        # Set to recommended gain
        self.sdr.gain = int(best_gain["gain"] * 10)
        return best_gain["gain"]

    def analyze_spectrum(self, duration=10):
        """Analyze the frequency spectrum around 1090 MHz"""
        print(f"\n📊 SPECTRUM ANALYSIS ({duration}s)")
        print("=" * 50)

        # Collect samples over time
        all_spectra = []
        frequencies = None

        for i in range(duration):
            samples = self.sdr.read_samples(1024 * 1024)

            # Compute power spectrum
            windowed = samples * signal.windows.hann(len(samples))
            fft_result = fft(windowed)
            power_linear = np.abs(fft_result) ** 2
            power_db = 10 * np.log10(power_linear + 1e-12)

            # Generate frequency array
            freqs = fftfreq(len(samples), 1 / self.sample_rate)
            freqs = np.fft.fftshift(freqs) + self.center_freq
            power_db = np.fft.fftshift(power_db)

            if frequencies is None:
                frequencies = freqs

            all_spectra.append(power_db)
            print(f"   Collected spectrum {i+1}/{duration}")
            time.sleep(1)

        # Average the spectra
        avg_spectrum = np.mean(all_spectra, axis=0)

        # Find peaks
        peaks, properties = signal.find_peaks(
            avg_spectrum,
            height=np.mean(avg_spectrum) + 10,  # 10 dB above average
            distance=int(len(avg_spectrum) * 0.01),
        )

        print(f"\n📈 SPECTRUM RESULTS:")
        print(
            f"   Frequency range: {frequencies[0]/1e6:.1f} - {frequencies[-1]/1e6:.1f} MHz"
        )
        print(f"   Average power: {np.mean(avg_spectrum):.1f} dB")
        print(f"   Peak power: {np.max(avg_spectrum):.1f} dB")
        print(f"   Signals detected: {len(peaks)}")

        # Report peaks near 1090 MHz
        for peak_idx in peaks:
            freq_mhz = frequencies[peak_idx] / 1e6
            power = avg_spectrum[peak_idx]
            if 1089 <= freq_mhz <= 1091:  # Within 1 MHz of 1090
                print(f"   🎯 Signal at {freq_mhz:.3f} MHz: {power:.1f} dB")

        return frequencies, avg_spectrum, peaks

    def detect_adsb_activity(self, duration=30):
        """Look for ADS-B-like signal activity"""
        print(f"\n🛩️  ADS-B ACTIVITY DETECTION ({duration}s)")
        print("=" * 50)

        pulse_count = 0
        strong_signals = 0
        sample_count = 0

        # Detection parameters
        pulse_threshold_factor = 5.0  # Pulses should be 5x above noise

        start_time = time.time()

        while time.time() - start_time < duration:
            samples = self.sdr.read_samples(256 * 1024)
            magnitude = np.abs(samples)
            sample_count += 1

            # Calculate dynamic threshold
            noise_floor = np.percentile(magnitude, 25)
            signal_peak = np.percentile(magnitude, 99)
            pulse_threshold = noise_floor + (signal_peak - noise_floor) * 0.8

            # Look for pulse-like activity
            above_threshold = magnitude > pulse_threshold

            # Count transitions (pulse edges)
            transitions = np.diff(above_threshold.astype(int))
            rising_edges = np.sum(transitions == 1)

            pulse_count += rising_edges

            # Count strong signals
            if signal_peak > noise_floor * pulse_threshold_factor:
                strong_signals += 1

            if sample_count % 30 == 0:  # Progress every ~3 seconds
                elapsed = time.time() - start_time
                print(
                    f"   {elapsed:.1f}s: pulses={pulse_count}, strong_signals={strong_signals}"
                )

        print(f"\n📊 ACTIVITY RESULTS:")
        print(f"   Total pulses detected: {pulse_count}")
        print(f"   Strong signal batches: {strong_signals}/{sample_count}")
        print(f"   Pulse rate: {pulse_count/duration:.1f} pulses/second")

        # Interpret results
        if pulse_count > 100:
            print("   ✅ Good pulse activity - likely ADS-B signals present")
        elif pulse_count > 20:
            print("   ⚠️  Some pulse activity - weak ADS-B signals or interference")
        else:
            print("   ❌ Very low pulse activity - check antenna and location")

        return pulse_count, strong_signals

    def check_antenna_connection(self):
        """Check for antenna connection issues"""
        print("\n🔌 ANTENNA CONNECTION CHECK")
        print("=" * 50)

        # Test with different gains to see if signal changes appropriately
        test_gains = [0, 20, 40]  # Low, medium, high gain
        power_levels = []

        for gain_db in test_gains:
            self.sdr.gain = gain_db * 10  # RTL-SDR uses tenths of dB
            time.sleep(0.2)

            samples = self.sdr.read_samples(256 * 1024)
            power = np.mean(np.abs(samples))
            power_levels.append(power)

            print(f"   Gain {gain_db:2d} dB: Power level {power:.6f}")

        # Check if power increases with gain (indicates antenna connected)
        power_increase = (
            power_levels[-1] / power_levels[0] if power_levels[0] > 0 else 0
        )

        print(f"\n📊 CONNECTION ANALYSIS:")
        print(f"   Power ratio (high/low gain): {power_increase:.1f}")

        if power_increase > 2.0:
            print("   ✅ Antenna appears to be connected (power increases with gain)")
        elif power_increase > 1.2:
            print("   ⚠️  Weak antenna connection or poor signal environment")
        else:
            print(
                "   ❌ Possible antenna connection issue (power doesn't increase with gain)"
            )

        return power_increase

    def run_full_analysis(self):
        """Run complete signal analysis"""
        print("🔍 ADS-B SIGNAL ANALYZER")
        print("=" * 60)

        if not self.connect_sdr():
            return

        try:
            # 1. Check antenna connection
            antenna_ratio = self.check_antenna_connection()

            # 2. Test gain levels
            optimal_gain = self.test_gain_levels()

            # 3. Analyze spectrum
            frequencies, spectrum, peaks = self.analyze_spectrum(duration=5)

            # 4. Detect ADS-B activity
            pulse_count, strong_signals = self.detect_adsb_activity(duration=15)

            # 5. Generate recommendations
            print("\n💡 RECOMMENDATIONS")
            print("=" * 50)

            if antenna_ratio < 1.5:
                print(
                    "🔧 Check antenna connection - signal doesn't increase properly with gain"
                )
                print("   - Ensure antenna is properly connected")
                print("   - Check coax cable for damage")
                print("   - Try a different antenna")

            if pulse_count < 20:
                print("📍 Low signal activity detected:")
                print("   - Move antenna to higher location")
                print("   - Check if you're in an area with air traffic")
                print("   - Try different antenna orientation")
                print("   - Consider a better antenna (1090 MHz optimized)")

            if len(peaks) == 0:
                print("📡 No strong signals detected:")
                print("   - Increase gain (try 40+ dB)")
                print("   - Check frequency calibration")
                print("   - Verify 1090 MHz is clear of interference")

            print(f"\n🎯 OPTIMAL SETTINGS:")
            print(f"   Gain: {optimal_gain:.1f} dB")
            print(f"   Sample rate: {self.sample_rate/1e6:.1f} MHz")
            print(f"   Frequency: {self.center_freq/1e6:.1f} MHz")

        except KeyboardInterrupt:
            print("\n🛑 Analysis interrupted by user")
        finally:
            if self.sdr:
                self.sdr.close()
            print("\n✅ Analysis completed")


def main():
    parser = argparse.ArgumentParser(description="ADS-B Signal Analyzer")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=2.0,
        help="Sample rate in MHz (default: 2.0)",
    )
    parser.add_argument(
        "--device-index", type=int, default=0, help="RTL-SDR device index (default: 0)"
    )

    args = parser.parse_args()

    analyzer = ADSBSignalAnalyzer(sample_rate=args.sample_rate * 1e6)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
