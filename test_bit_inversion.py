#!/usr/bin/env python3
"""
Test bit inversion to see if that fixes CRC issues
"""

import numpy as np
from rtlsdr import RtlSdr
import time

def calculate_crc(data: bytes) -> int:
    """Calculate CRC-24 for Mode S messages"""
    # CRC polynomial for Mode S: 0x1FFF409 (CRC-24)
    polynomial = 0x1FFF409
    crc = 0

    for byte in data[:-3]:  # Exclude last 3 bytes (CRC field)
        crc ^= byte << 16
        for _ in range(8):
            if crc & 0x800000:
                crc = (crc << 1) ^ polynomial
            else:
                crc <<= 1
            crc &= 0xFFFFFF

    return crc

def bits_to_bytes(bits):
    """Convert bit array to bytes"""
    if len(bits) % 8 != 0:
        return None

    bytes_data = bytearray()
    for i in range(0, len(bits), 8):
        byte_val = 0
        for j in range(8):
            if i + j < len(bits):
                byte_val |= bits[i + j] << (7 - j)
        bytes_data.append(byte_val)

    return bytes(bytes_data)

def test_bit_inversion():
    """Test if bit inversion helps with CRC"""

    # Connect to SDR
    sdr = RtlSdr()
    sdr.sample_rate = 2.0e6
    sdr.center_freq = 1090e6
    sdr.gain = 'auto'

    print("Testing bit inversion for CRC fixes...")
    print("Looking for messages with valid CRC after inversion...")

    try:
        for test_num in range(20):
            # Get samples
            samples = sdr.read_samples(256*1024)
            magnitude = np.abs(samples)

            # Simple threshold
            threshold = np.mean(magnitude) + 1.5 * np.std(magnitude)

            # Look for preamble pattern (simplified)
            preamble_pattern = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
            samples_per_bit = 2.0

            for i in range(len(magnitude) - 300):
                # Check for preamble
                preamble_match = True
                for j, expected_bit in enumerate(preamble_pattern):
                    sample_idx = int(i + j * samples_per_bit)
                    if sample_idx >= len(magnitude):
                        preamble_match = False
                        break
                    actual_bit = 1 if magnitude[sample_idx] > threshold else 0
                    if actual_bit != expected_bit:
                        preamble_match = False
                        break

                if preamble_match:
                    # Extract message bits
                    message_start = i + len(preamble_pattern) * samples_per_bit
                    bits = []

                    for bit_idx in range(112):
                        sample_idx = int(message_start + bit_idx * samples_per_bit)
                        if sample_idx >= len(magnitude):
                            break
                        bit_value = 1 if magnitude[sample_idx] > threshold else 0
                        bits.append(bit_value)

                    if len(bits) == 112:
                        # Test normal bits
                        message_bytes = bits_to_bytes(bits)
                        if message_bytes:
                            calc_crc = calculate_crc(message_bytes)
                            recv_crc = int.from_bytes(message_bytes[-3:], 'big')

                            if calc_crc == recv_crc:
                                hex_data = ' '.join(f'{b:02X}' for b in message_bytes)
                                print(f"✅ NORMAL BITS VALID CRC: {hex_data}")
                                print(f"   CRC: {calc_crc:06X}")
                                return True

                        # Test inverted bits
                        inverted_bits = [1 - bit for bit in bits]
                        message_bytes = bits_to_bytes(inverted_bits)
                        if message_bytes:
                            calc_crc = calculate_crc(message_bytes)
                            recv_crc = int.from_bytes(message_bytes[-3:], 'big')

                            if calc_crc == recv_crc:
                                hex_data = ' '.join(f'{b:02X}' for b in message_bytes)
                                print(f"✅ INVERTED BITS VALID CRC: {hex_data}")
                                print(f"   CRC: {calc_crc:06X}")
                                return True

                            # Show some examples for debugging
                            if test_num < 3:
                                hex_data = ' '.join(f'{b:02X}' for b in message_bytes)
                                print(f"❌ Test {test_num}: {hex_data}")
                                print(f"   Calc CRC: {calc_crc:06X}, Recv CRC: {recv_crc:06X}")

            print(f"Test {test_num + 1}/20 completed")
            time.sleep(0.1)

    finally:
        sdr.close()

    print("No valid CRC found with normal or inverted bits")
    return False

if __name__ == "__main__":
    test_bit_inversion()
