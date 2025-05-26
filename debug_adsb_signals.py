#!/usr/bin/env python3
"""
Debug ADS-B Signal Processing
Simple command-line tool to debug ADS-B reception issues
"""

import numpy as np
import time
import argparse
from adsb_scanner import ADSBScanner

def main():
    parser = argparse.ArgumentParser(description='Debug ADS-B Signal Processing')
    parser.add_argument('--sample-rate', type=float, default=2.0,
                       help='Sample rate in MHz (default: 2.0)')
    parser.add_argument('--device-index', type=int, default=0,
                       help='RTL-SDR device index (default: 0)')
    parser.add_argument('--duration', type=int, default=30,
                       help='Duration to run in seconds (default: 30)')

    args = parser.parse_args()

    print("🐛 ADS-B Signal Debug Tool")
    print("=" * 50)
    print(f"Sample rate: {args.sample_rate} MHz")
    print(f"Duration: {args.duration} seconds")
    print(f"Debug logging: ENABLED")
    print()

    # Create scanner with debug enabled
    scanner = ADSBScanner(sample_rate=args.sample_rate * 1e6, debug=True)

    # Connect to SDR
    if not scanner.connect_sdr(args.device_index):
        print("❌ Failed to connect to SDR device")
        return

    print("✅ Connected to SDR device")
    print("🔍 Starting signal analysis...")
    print()

    start_time = time.time()
    sample_count = 0

    try:
        while time.time() - start_time < args.duration:
            # Collect samples
            samples = scanner.sdr.read_samples(256*1024)
            sample_count += 1

            # Process samples
            scanner.process_samples(samples)

            # Print progress every 5 seconds
            elapsed = time.time() - start_time
            if sample_count % 50 == 0:  # Roughly every 5 seconds
                print(f"⏱️  {elapsed:.1f}s - Processed {sample_count} sample batches")
                print(f"   Total messages: {scanner.total_messages}")
                print(f"   Valid messages: {scanner.valid_messages}")
                if scanner.total_messages > 0:
                    success_rate = (scanner.valid_messages / scanner.total_messages) * 100
                    print(f"   Success rate: {success_rate:.1f}%")
                print(f"   Aircraft detected: {len(scanner.aircraft)}")
                print()

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")

    finally:
        # Print final statistics
        elapsed = time.time() - start_time
        print("\n" + "=" * 50)
        print("📊 FINAL STATISTICS")
        print("=" * 50)
        print(f"Runtime: {elapsed:.1f} seconds")
        print(f"Sample batches processed: {sample_count}")
        print(f"Total messages detected: {scanner.total_messages}")
        print(f"Valid messages decoded: {scanner.valid_messages}")

        if scanner.total_messages > 0:
            success_rate = (scanner.valid_messages / scanner.total_messages) * 100
            print(f"Overall success rate: {success_rate:.1f}%")
        else:
            print("No messages detected - check antenna and location")

        print(f"Aircraft detected: {len(scanner.aircraft)}")

        # Print debug statistics
        if scanner.debug:
            scanner.print_debug_stats()

        # Print aircraft if any were detected
        if scanner.aircraft:
            print("\n🛩️  DETECTED AIRCRAFT:")
            scanner.print_aircraft_table()

        scanner.disconnect_sdr()
        print("\n✅ Debug session completed")

if __name__ == "__main__":
    main()
