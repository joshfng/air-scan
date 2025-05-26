#!/usr/bin/env python3
"""
ADS-B Aircraft Tracker - 1090 MHz Mode S transponder decoder
Tracks aircraft using ADS-B signals at 1090 MHz
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import threading
import time
import argparse
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import sys
import struct
import math
import logging

try:
    from rtlsdr import RtlSdr
except ImportError:
    print("pyrtlsdr not installed. Install with: pip install pyrtlsdr")
    sys.exit(1)

from scipy import signal


@dataclass
class Aircraft:
    """Represents a tracked aircraft"""

    icao: str
    callsign: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[int] = None
    velocity: Optional[float] = None
    heading: Optional[float] = None
    vertical_rate: Optional[int] = None
    squawk: Optional[str] = None
    last_seen: float = field(default_factory=time.time)
    message_count: int = 0
    position_messages: int = 0


@dataclass
class ADSBMessage:
    """Represents a decoded ADS-B message"""

    timestamp: float
    icao: str
    message_type: int
    raw_data: bytes
    decoded_data: Dict = field(default_factory=dict)


class ADSBDecoder:
    """ADS-B Mode S message decoder"""

    def __init__(self, debug=False):
        # CRC polynomial for Mode S
        self.crc_poly = 0xFFF409
        self.debug = debug

        if self.debug:
            self.logger = logging.getLogger("ADSB.Decoder")
        else:
            self.logger = None

        # Message type definitions
        self.message_types = {
            1: "Aircraft Identification and Category",
            2: "Aircraft Identification and Category",
            3: "Aircraft Identification and Category",
            4: "Aircraft Identification and Category",
            9: "Airborne Position (Baro Altitude)",
            10: "Airborne Position (Baro Altitude)",
            11: "Airborne Position (Baro Altitude)",
            18: "Airborne Position (GNSS Height)",
            19: "Airborne Velocity",
            20: "Airborne Position (GNSS Height)",
            21: "Airborne Position (GNSS Height)",
            22: "Airborne Position (GNSS Height)",
        }

    def calculate_crc(self, data: bytes) -> int:
        """Calculate CRC-24 for Mode S message"""
        crc = 0
        for byte in data[:-3]:  # Exclude last 3 bytes (CRC field)
            crc ^= byte << 16
            for _ in range(8):
                if crc & 0x800000:
                    crc = (crc << 1) ^ self.crc_poly
                else:
                    crc <<= 1
                crc &= 0xFFFFFF
        return crc

    def verify_message(self, data: bytes) -> bool:
        """Verify Mode S message CRC"""
        if len(data) < 14:
            if self.debug and self.logger:
                self.logger.debug(f"Message too short: {len(data)} bytes")
            return False

        calculated_crc = self.calculate_crc(data)
        received_crc = (data[-3] << 16) | (data[-2] << 8) | data[-1]
        is_valid = calculated_crc == received_crc

        if self.debug and self.logger:
            if is_valid:
                self.logger.debug(
                    f"✓ CRC valid: calc={calculated_crc:06X}, recv={received_crc:06X}"
                )
            else:
                self.logger.debug(
                    f"✗ CRC invalid: calc={calculated_crc:06X}, recv={received_crc:06X}"
                )

        return is_valid

    def decode_icao(self, data: bytes) -> str:
        """Extract ICAO address from message"""
        if len(data) < 7:
            return ""
        icao = (data[1] << 16) | (data[2] << 8) | data[3]
        return f"{icao:06X}"

    def decode_callsign(self, data: bytes) -> str:
        """Decode aircraft callsign from identification message"""
        if len(data) < 14:
            return ""

        # Character set for callsign decoding
        charset = "?ABCDEFGHIJKLMNOPQRSTUVWXYZ????? ???????????????0123456789??????"

        callsign = ""
        # Extract 6-bit characters from message
        for i in range(8):
            if i < 6:
                char_code = (data[5 + i // 2] >> (4 - 4 * (i % 2))) & 0x3F
                if char_code < len(charset):
                    callsign += charset[char_code]

        return callsign.strip()

    def decode_altitude(self, data: bytes) -> Optional[int]:
        """Decode altitude from airborne position message"""
        if len(data) < 14:
            return None

        # Extract altitude bits
        alt_bits = ((data[5] & 0xFF) << 4) | ((data[6] & 0xF0) >> 4)

        if alt_bits == 0:
            return None

        # Decode altitude (simplified - assumes 25ft increments)
        altitude = (alt_bits - 1) * 25
        return altitude if altitude >= 0 else None

    def decode_position(self, data: bytes) -> Tuple[Optional[float], Optional[float]]:
        """Decode latitude/longitude from position message (simplified)"""
        if len(data) < 14:
            return None, None

        # This is a simplified decoder - full CPR decoding requires multiple messages
        # Extract raw position bits
        lat_cpr = ((data[6] & 0x03) << 15) | (data[7] << 7) | ((data[8] & 0xFE) >> 1)
        lon_cpr = ((data[8] & 0x01) << 16) | (data[9] << 8) | data[10]

        # For demonstration, return normalized values
        # Real implementation would need CPR decoding with reference position
        lat = (lat_cpr / 131072.0) * 180.0 - 90.0
        lon = (lon_cpr / 131072.0) * 360.0 - 180.0

        return lat, lon

    def decode_velocity(
        self, data: bytes
    ) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        """Decode velocity and heading from velocity message"""
        if len(data) < 14:
            return None, None, None

        # Extract velocity components
        ew_vel = ((data[5] & 0x03) << 8) | data[6]
        ns_vel = ((data[7] & 0xFF) << 2) | ((data[8] & 0xC0) >> 6)

        # Calculate ground speed and heading
        if ew_vel != 0 or ns_vel != 0:
            # Convert to actual velocity (simplified)
            ew_vel = ew_vel - 1 if ew_vel > 0 else 0
            ns_vel = ns_vel - 1 if ns_vel > 0 else 0

            velocity = math.sqrt(ew_vel**2 + ns_vel**2)
            heading = math.degrees(math.atan2(ew_vel, ns_vel))
            if heading < 0:
                heading += 360

            # Extract vertical rate
            vr_bits = ((data[8] & 0x07) << 6) | ((data[9] & 0xFC) >> 2)
            vertical_rate = (vr_bits - 1) * 64 if vr_bits > 0 else 0

            return velocity, heading, vertical_rate

        return None, None, None

    def decode_message(self, data: bytes) -> Optional[ADSBMessage]:
        """Decode a complete ADS-B message"""
        if not self.verify_message(data):
            return None

        # Extract message type
        df = (data[0] & 0xF8) >> 3  # Downlink Format
        if df != 17:  # ADS-B messages are DF=17
            return None

        type_code = (data[4] & 0xF8) >> 3
        icao = self.decode_icao(data)

        message = ADSBMessage(
            timestamp=time.time(), icao=icao, message_type=type_code, raw_data=data
        )

        # Decode based on message type
        if 1 <= type_code <= 4:  # Aircraft identification
            message.decoded_data["callsign"] = self.decode_callsign(data)
        elif 9 <= type_code <= 18:  # Airborne position
            message.decoded_data["altitude"] = self.decode_altitude(data)
            lat, lon = self.decode_position(data)
            message.decoded_data["latitude"] = lat
            message.decoded_data["longitude"] = lon
        elif type_code == 19:  # Velocity
            vel, heading, vr = self.decode_velocity(data)
            message.decoded_data["velocity"] = vel
            message.decoded_data["heading"] = heading
            message.decoded_data["vertical_rate"] = vr

        return message


class ADSBScanner:
    """ADS-B aircraft tracker using RTL-SDR"""

    def __init__(self, sample_rate=2.0e6, debug=False):
        self.sample_rate = sample_rate
        self.center_freq = 1090e6  # ADS-B frequency
        self.debug = debug

        # Improved signal processing parameters
        self.min_preamble_snr = 6.0  # Minimum SNR for preamble detection
        self.bit_sync_tolerance = 0.3  # Tolerance for bit timing sync

        # Setup debug logging
        if self.debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler("adsb_debug.log"),
                ],
            )
            self.logger = logging.getLogger("ADSB")
            self.logger.info("ADS-B Debug logging enabled")
        else:
            self.logger = logging.getLogger("ADSB")
            self.logger.setLevel(logging.WARNING)

        # SDR device
        self.sdr = None
        self.scanning = False

        # Message processing
        self.decoder = ADSBDecoder(debug=self.debug)
        self.aircraft = {}  # ICAO -> Aircraft
        self.messages = deque(maxlen=1000)

        # Signal detection parameters
        self.preamble = np.array(
            [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
        )  # Mode S preamble
        self.message_length = 112  # Mode S short message length in bits
        self.long_message_length = 224  # Mode S long message length in bits

        # Statistics
        self.total_messages = 0
        self.valid_messages = 0
        self.aircraft_count = 0
        self.debug_stats = {
            "preambles_detected": 0,
            "crc_failures": 0,
            "decode_failures": 0,
            "message_types": {},
            "signal_strength_samples": [],
        }

        # Plotting
        self.fig = None
        self.ax_map = None
        self.ax_stats = None

    def connect_sdr(self, device_index=0, gain="auto"):
        """Connect to RTL-SDR device"""
        try:
            self.sdr = RtlSdr(device_index)
            self.sdr.sample_rate = self.sample_rate
            self.sdr.center_freq = self.center_freq
            self.sdr.gain = gain  # Use automatic gain control for better resolution
            print(f"Connected to SDR device {device_index}")
            print(f"Sample rate: {self.sample_rate/1e6:.1f} MHz")
            print(f"Center frequency: {self.center_freq/1e6:.1f} MHz")
            if gain == "auto":
                print(f"Gain: Automatic")
            else:
                print(f"Gain: {self.sdr.gain} dB")
            return True
        except Exception as e:
            print(f"Failed to connect to SDR device: {e}")
            return False

    def disconnect_sdr(self):
        """Disconnect from SDR device"""
        if self.sdr:
            self.sdr.close()
            self.sdr = None
            print("SDR device disconnected")

    def detect_preambles(self, samples):
        """Detect Mode S preambles using simple but robust edge detection"""
        # Convert to magnitude
        magnitude = np.abs(samples)

        # Use high threshold to detect strong Mode S signals - 99th percentile
        threshold = np.percentile(magnitude, 99.0)

        if self.debug:
            self.logger.debug(
                f"Signal analysis: mean={np.mean(magnitude):.3f}, "
                f"std={np.std(magnitude):.3f}, threshold={threshold:.3f}"
            )
            self.debug_stats["signal_strength_samples"].append(np.mean(magnitude))
            if len(self.debug_stats["signal_strength_samples"]) > 100:
                self.debug_stats["signal_strength_samples"].pop(0)

        # Mode S preamble: 1010000010100000 (16 bits at 1 MHz = 16 μs)
        # At 2 MHz sampling: 32 samples for preamble
        samples_per_bit = 2
        preamble_bits = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]

        preamble_locations = []

        # Simple pattern matching - look for the preamble pattern
        for i in range(len(magnitude) - 32):
            if magnitude[i] > threshold:  # Potential start of preamble
                # Check preamble pattern
                valid_preamble = True
                confidence = 0

                for bit_idx, expected_bit in enumerate(preamble_bits):
                    sample_idx = i + bit_idx * samples_per_bit

                    if sample_idx + 1 < len(magnitude):
                        # Average the samples for this bit
                        bit_samples = magnitude[
                            sample_idx : sample_idx + samples_per_bit
                        ]
                        bit_value = 1 if np.mean(bit_samples) > threshold else 0

                        if bit_value == expected_bit:
                            confidence += 1
                        else:
                            # Allow some tolerance for noise
                            if confidence < bit_idx * 0.7:  # Need 70% match so far
                                valid_preamble = False
                                break

                # Require at least 80% match
                if valid_preamble and confidence >= len(preamble_bits) * 0.8:
                    # Message starts immediately after the preamble
                    message_start = i + len(preamble_bits) * samples_per_bit
                    preamble_locations.append(message_start)

                    if self.debug:
                        self.logger.debug(
                            f"Preamble found at sample {i}, "
                            f"confidence: {confidence}/{len(preamble_bits)}, "
                            f"message starts at: {message_start}"
                        )

                    # Skip ahead to avoid detecting overlapping preambles
                    break  # Process one preamble at a time for better accuracy

        if self.debug and preamble_locations:
            self.debug_stats["preambles_detected"] += len(preamble_locations)
            self.logger.debug(
                f"Found {len(preamble_locations)} preambles in this batch"
            )

        return preamble_locations

    def extract_message_bits(self, samples, start_idx, message_bits=112):
        """Extract message bits from samples using improved bit recovery"""
        bits = []
        samples_per_bit = (
            self.sample_rate / 1e6
        )  # 1 MHz bit rate (use float for precision)

        magnitude = np.abs(samples)

        # Use high threshold based on signal analysis - 99th percentile
        threshold = np.percentile(magnitude, 99.0)  # Use 99th percentile as threshold

        if self.debug:
            self.logger.debug(
                f"Bit extraction: start_idx={start_idx}, "
                f"samples_per_bit={samples_per_bit:.1f}, threshold={threshold:.3f}"
            )

        for i in range(message_bits):
            # Sample at multiple points within each bit period and take the maximum
            bit_start = start_idx + i * samples_per_bit
            bit_end = start_idx + (i + 1) * samples_per_bit

            # Sample at 1/4, 1/2, and 3/4 points within the bit period
            sample_points = [
                int(bit_start + 0.25 * samples_per_bit),
                int(bit_start + 0.5 * samples_per_bit),
                int(bit_start + 0.75 * samples_per_bit),
            ]

            # Take the maximum value within the bit period
            bit_values = []
            for point in sample_points:
                if point < len(magnitude):
                    bit_values.append(magnitude[point])

            if not bit_values:
                if self.debug:
                    self.logger.debug(
                        f"Bit extraction truncated at bit {i}/{message_bits}"
                    )
                break

            bit_value = max(bit_values)
            bits.append(1 if bit_value > threshold else 0)

        if self.debug and len(bits) > 0:
            bit_string = "".join(str(b) for b in bits[:32])  # First 32 bits for debug
            self.logger.debug(f"Extracted {len(bits)} bits, first 32: {bit_string}")

        return bits

    def bits_to_bytes(self, bits):
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

    def validate_message_structure(self, bits):
        """Validate basic message structure before CRC check"""
        if len(bits) < 112:
            return False

        # Check downlink format (first 5 bits)
        df_bits = bits[:5]
        df_value = sum(bit * (2 ** (4 - i)) for i, bit in enumerate(df_bits))

        if self.debug:
            self.logger.debug(f"DF: {df_value}")

        # Accept common Mode S formats:
        # DF=17: ADS-B Extended Squitter (preferred)
        # DF=18: TIS-B Extended Squitter
        # DF=4,5: Surveillance Altitude Reply
        # DF=20,21: Comm-B Altitude Reply
        # DF=11: All-Call Reply
        # DF=0: Short Air-Air Surveillance
        valid_dfs = [0, 4, 5, 11, 17, 18, 20, 21]

        if df_value not in valid_dfs:
            if self.debug:
                self.logger.debug(f"Unsupported DF: {df_value}")
            return False

        # For DF=17/18, check capability field (bits 5-7)
        if df_value in [17, 18]:
            ca_bits = bits[5:8]
            ca_value = sum(bit * (2 ** (2 - i)) for i, bit in enumerate(ca_bits))

            if ca_value > 7:  # CA field is 3 bits, max value 7
                if self.debug:
                    self.logger.debug(f"Invalid CA: {ca_value}")
                return False

        return True

    def process_samples(self, samples):
        """Process samples to extract ADS-B messages"""
        preamble_locations = self.detect_preambles(samples)

        for location in preamble_locations:
            self.total_messages += 1

            # Try short message first (112 bits)
            bits = self.extract_message_bits(samples, location, 112)
            if len(bits) == 112:
                # Validate message structure before attempting decode
                if not self.validate_message_structure(bits):
                    if self.debug:
                        self.debug_stats["decode_failures"] += 1
                        self.logger.debug("✗ Message structure validation failed")
                    continue

                message_bytes = self.bits_to_bytes(bits)
                if message_bytes:
                    if self.debug:
                        hex_data = " ".join(f"{b:02X}" for b in message_bytes)
                        self.logger.debug(
                            f"Attempting to decode 112-bit message: {hex_data}"
                        )

                    message = self.decoder.decode_message(message_bytes)
                    if message:
                        self.valid_messages += 1
                        self.messages.append(message)
                        self.update_aircraft(message)

                        if self.debug:
                            self.logger.debug(
                                f"✓ Successfully decoded message from {message.icao}, "
                                f"type {message.message_type}: {message.decoded_data}"
                            )
                            msg_type = message.message_type
                            self.debug_stats["message_types"][msg_type] = (
                                self.debug_stats["message_types"].get(msg_type, 0) + 1
                            )
                        continue
                    else:
                        if self.debug:
                            self.debug_stats["decode_failures"] += 1
                            self.debug_stats["crc_failures"] += 1
                            self.logger.debug(
                                "✗ Message decode failed (CRC or format error)"
                            )

            # Try long message (224 bits)
            bits = self.extract_message_bits(samples, location, 224)
            if len(bits) == 224:
                message_bytes = self.bits_to_bytes(bits)
                if message_bytes:
                    if self.debug:
                        hex_data = " ".join(f"{b:02X}" for b in message_bytes)
                        self.logger.debug(
                            f"Attempting to decode 224-bit message: {hex_data}"
                        )

                    message = self.decoder.decode_message(message_bytes)
                    if message:
                        self.valid_messages += 1
                        self.messages.append(message)
                        self.update_aircraft(message)

                        if self.debug:
                            self.logger.debug(
                                f"✓ Successfully decoded long message from {message.icao}, "
                                f"type {message.message_type}: {message.decoded_data}"
                            )
                            msg_type = message.message_type
                            self.debug_stats["message_types"][msg_type] = (
                                self.debug_stats["message_types"].get(msg_type, 0) + 1
                            )
                    else:
                        if self.debug:
                            self.debug_stats["decode_failures"] += 1
                            self.debug_stats["crc_failures"] += 1
                            self.logger.debug(
                                "✗ Long message decode failed (CRC or format error)"
                            )

    def update_aircraft(self, message: ADSBMessage):
        """Update aircraft database with new message"""
        icao = message.icao

        is_new_aircraft = icao not in self.aircraft
        if is_new_aircraft:
            self.aircraft[icao] = Aircraft(icao=icao)
            self.aircraft_count += 1
            if self.debug:
                self.logger.info(f"🛩️  NEW AIRCRAFT: {icao}")

        aircraft = self.aircraft[icao]
        aircraft.last_seen = message.timestamp
        aircraft.message_count += 1

        # Track what data was updated
        updates = []

        # Update aircraft data based on message content
        if "callsign" in message.decoded_data:
            old_callsign = aircraft.callsign
            aircraft.callsign = message.decoded_data["callsign"]
            if old_callsign != aircraft.callsign:
                updates.append(f"callsign: {aircraft.callsign}")

        if "altitude" in message.decoded_data:
            old_altitude = aircraft.altitude
            aircraft.altitude = message.decoded_data["altitude"]
            if old_altitude != aircraft.altitude:
                updates.append(f"altitude: {aircraft.altitude}ft")

        if "latitude" in message.decoded_data and "longitude" in message.decoded_data:
            old_lat, old_lon = aircraft.latitude, aircraft.longitude
            aircraft.latitude = message.decoded_data["latitude"]
            aircraft.longitude = message.decoded_data["longitude"]
            aircraft.position_messages += 1
            if old_lat != aircraft.latitude or old_lon != aircraft.longitude:
                updates.append(
                    f"position: ({aircraft.latitude:.4f}, {aircraft.longitude:.4f})"
                )

        if "velocity" in message.decoded_data:
            old_velocity = aircraft.velocity
            aircraft.velocity = message.decoded_data["velocity"]
            if old_velocity != aircraft.velocity:
                updates.append(f"velocity: {aircraft.velocity:.0f}kt")

        if "heading" in message.decoded_data:
            old_heading = aircraft.heading
            aircraft.heading = message.decoded_data["heading"]
            if old_heading != aircraft.heading:
                updates.append(f"heading: {aircraft.heading:.0f}°")

        if "vertical_rate" in message.decoded_data:
            old_vr = aircraft.vertical_rate
            aircraft.vertical_rate = message.decoded_data["vertical_rate"]
            if old_vr != aircraft.vertical_rate:
                updates.append(f"vertical_rate: {aircraft.vertical_rate}fpm")

        if self.debug and updates:
            self.logger.info(f"📡 {icao} updated: {', '.join(updates)}")

    def cleanup_old_aircraft(self, max_age=300):
        """Remove aircraft not seen for max_age seconds"""
        current_time = time.time()
        to_remove = []

        for icao, aircraft in self.aircraft.items():
            if current_time - aircraft.last_seen > max_age:
                to_remove.append(icao)

        for icao in to_remove:
            del self.aircraft[icao]
            self.aircraft_count -= 1

    def scan_adsb(self):
        """Main ADS-B scanning loop"""
        print("Starting ADS-B aircraft tracking...")
        print("Listening for Mode S transponder signals at 1090 MHz...")

        while self.scanning:
            try:
                # Collect samples
                samples = self.sdr.read_samples(256 * 1024)

                # Process for ADS-B messages
                self.process_samples(samples)

                # Cleanup old aircraft every 30 seconds
                if self.total_messages % 1000 == 0:
                    self.cleanup_old_aircraft()

                # Print statistics every 5000 messages
                if self.total_messages % 5000 == 0 and self.total_messages > 0:
                    success_rate = (self.valid_messages / self.total_messages) * 100
                    print(
                        f"Messages: {self.total_messages}, Valid: {self.valid_messages} "
                        f"({success_rate:.1f}%), Aircraft: {len(self.aircraft)}"
                    )

                    if self.debug:
                        self.print_debug_stats()

                time.sleep(0.01)  # Small delay to prevent overwhelming

            except Exception as e:
                print(f"Error during ADS-B scanning: {e}")
                break

    def setup_plot(self):
        """Setup real-time plotting"""
        plt.style.use("dark_background")
        self.fig, (self.ax_map, self.ax_stats) = plt.subplots(1, 2, figsize=(16, 8))

        # Map plot
        self.ax_map.set_title("Aircraft Positions", fontsize=14, fontweight="bold")
        self.ax_map.set_xlabel("Longitude")
        self.ax_map.set_ylabel("Latitude")
        self.ax_map.grid(True, alpha=0.3)
        self.ax_map.set_facecolor("black")

        # Statistics plot
        self.ax_stats.set_title("ADS-B Statistics", fontsize=14, fontweight="bold")
        self.ax_stats.set_facecolor("black")

        plt.tight_layout()

        # Add control buttons
        ax_start = plt.axes([0.1, 0.02, 0.1, 0.04])
        ax_stop = plt.axes([0.25, 0.02, 0.1, 0.04])
        ax_clear = plt.axes([0.4, 0.02, 0.1, 0.04])

        self.btn_start = Button(ax_start, "Start")
        self.btn_stop = Button(ax_stop, "Stop")
        self.btn_clear = Button(ax_clear, "Clear")

        self.btn_start.on_clicked(self.start_scanning)
        self.btn_stop.on_clicked(self.stop_scanning)
        self.btn_clear.on_clicked(self.clear_aircraft)

    def start_scanning(self, event=None):
        """Start scanning in a separate thread"""
        if not self.scanning:
            self.scanning = True
            self.scan_thread = threading.Thread(target=self.scan_adsb)
            self.scan_thread.daemon = True
            self.scan_thread.start()
            print("ADS-B scanning started")

    def stop_scanning(self, event=None):
        """Stop scanning"""
        self.scanning = False
        print("ADS-B scanning stopped")

    def clear_aircraft(self, event=None):
        """Clear aircraft database"""
        self.aircraft.clear()
        self.messages.clear()
        self.total_messages = 0
        self.valid_messages = 0
        self.aircraft_count = 0
        print("Aircraft database cleared")

    def update_plot(self, frame):
        """Update plot animation"""
        # Clear plots
        self.ax_map.clear()
        self.ax_stats.clear()

        # Plot aircraft positions
        current_time = time.time()
        active_aircraft = []

        for aircraft in self.aircraft.values():
            if current_time - aircraft.last_seen < 60:  # Active in last minute
                active_aircraft.append(aircraft)

                if aircraft.latitude is not None and aircraft.longitude is not None:
                    # Plot aircraft position
                    self.ax_map.scatter(
                        aircraft.longitude, aircraft.latitude, c="red", s=50, alpha=0.8
                    )

                    # Add aircraft label
                    label = aircraft.callsign if aircraft.callsign else aircraft.icao
                    if aircraft.altitude:
                        label += f"\n{aircraft.altitude}ft"

                    self.ax_map.annotate(
                        label,
                        (aircraft.longitude, aircraft.latitude),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        color="yellow",
                    )

        # Update map
        self.ax_map.set_title(
            f"Aircraft Positions ({len(active_aircraft)} active)",
            fontsize=14,
            fontweight="bold",
        )
        self.ax_map.set_xlabel("Longitude")
        self.ax_map.set_ylabel("Latitude")
        self.ax_map.grid(True, alpha=0.3)
        self.ax_map.set_facecolor("black")

        # Update statistics
        stats_text = f"Total Messages: {self.total_messages}\n"
        stats_text += f"Valid Messages: {self.valid_messages}\n"
        if self.total_messages > 0:
            success_rate = (self.valid_messages / self.total_messages) * 100
            stats_text += f"Success Rate: {success_rate:.1f}%\n"
        stats_text += f"Total Aircraft: {len(self.aircraft)}\n"
        stats_text += f"Active Aircraft: {len(active_aircraft)}\n\n"

        # Add aircraft details
        stats_text += "Recent Aircraft:\n"
        for aircraft in sorted(
            active_aircraft, key=lambda x: x.last_seen, reverse=True
        )[:10]:
            age = current_time - aircraft.last_seen
            callsign = aircraft.callsign if aircraft.callsign else "Unknown"
            stats_text += f"{aircraft.icao}: {callsign} ({age:.0f}s ago)\n"
            if aircraft.altitude:
                stats_text += f"  Alt: {aircraft.altitude}ft"
            if aircraft.velocity:
                stats_text += f"  Spd: {aircraft.velocity:.0f}kt"
            if aircraft.heading:
                stats_text += f"  Hdg: {aircraft.heading:.0f}°"
            stats_text += "\n"

        self.ax_stats.text(
            0.05,
            0.95,
            stats_text,
            transform=self.ax_stats.transAxes,
            fontsize=10,
            verticalalignment="top",
            color="white",
            fontfamily="monospace",
        )
        self.ax_stats.set_title("ADS-B Statistics", fontsize=14, fontweight="bold")
        self.ax_stats.set_facecolor("black")
        self.ax_stats.set_xticks([])
        self.ax_stats.set_yticks([])

    def print_debug_stats(self):
        """Print detailed debug statistics"""
        if not self.debug:
            return

        self.logger.info("=== DEBUG STATISTICS ===")
        self.logger.info(
            f"Preambles detected: {self.debug_stats['preambles_detected']}"
        )
        self.logger.info(f"CRC failures: {self.debug_stats['crc_failures']}")
        self.logger.info(f"Decode failures: {self.debug_stats['decode_failures']}")

        if self.debug_stats["message_types"]:
            self.logger.info("Message types received:")
            for msg_type, count in sorted(self.debug_stats["message_types"].items()):
                type_name = self.decoder.message_types.get(
                    msg_type, f"Unknown({msg_type})"
                )
                self.logger.info(f"  Type {msg_type}: {count} messages ({type_name})")

        if self.debug_stats["signal_strength_samples"]:
            avg_signal = np.mean(self.debug_stats["signal_strength_samples"])
            self.logger.info(f"Average signal strength: {avg_signal:.3f}")

        self.logger.info("========================")

    def print_aircraft_table(self):
        """Print table of tracked aircraft"""
        if not self.aircraft:
            print("\nNo aircraft tracked")
            return

        print("\n" + "=" * 100)
        print("TRACKED AIRCRAFT")
        print("=" * 100)
        print(
            f"{'ICAO':<8} {'Callsign':<10} {'Lat':<10} {'Lon':<11} {'Alt(ft)':<8} "
            f"{'Spd(kt)':<8} {'Hdg':<6} {'Msgs':<6} {'Age(s)':<8}"
        )
        print("-" * 100)

        current_time = time.time()
        sorted_aircraft = sorted(
            self.aircraft.values(), key=lambda x: x.last_seen, reverse=True
        )

        for aircraft in sorted_aircraft[:20]:  # Show top 20
            age = current_time - aircraft.last_seen
            callsign = aircraft.callsign[:9] if aircraft.callsign else "Unknown"
            lat = f"{aircraft.latitude:.4f}" if aircraft.latitude else "Unknown"
            lon = f"{aircraft.longitude:.4f}" if aircraft.longitude else "Unknown"
            alt = str(aircraft.altitude) if aircraft.altitude else "Unknown"
            spd = f"{aircraft.velocity:.0f}" if aircraft.velocity else "Unknown"
            hdg = f"{aircraft.heading:.0f}°" if aircraft.heading else "Unknown"

            print(
                f"{aircraft.icao:<8} {callsign:<10} {lat:<10} {lon:<11} {alt:<8} "
                f"{spd:<8} {hdg:<6} {aircraft.message_count:<6} {age:<8.0f}"
            )

    def run(self):
        """Main run function"""
        gain = getattr(self, "gain", "auto")  # Default to auto if not set
        if not self.connect_sdr(gain=gain):
            return

        try:
            # Setup plotting
            self.setup_plot()

            # Start animation
            ani = animation.FuncAnimation(
                self.fig,
                self.update_plot,
                interval=2000,
                blit=False,
                cache_frame_data=False,
            )

            # Print aircraft table periodically
            def print_table():
                while True:
                    time.sleep(10)
                    if self.scanning:
                        self.print_aircraft_table()

            table_thread = threading.Thread(target=print_table)
            table_thread.daemon = True
            table_thread.start()

            print("\nADS-B Aircraft Tracker Ready!")
            print("Click 'Start' to begin tracking aircraft at 1090 MHz")
            print("Aircraft positions will appear on the map as they are detected")

            plt.show()

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.stop_scanning()
            self.disconnect_sdr()


def main():
    parser = argparse.ArgumentParser(
        description="ADS-B Aircraft Tracker - 1090 MHz Mode S decoder"
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=2.0,
        help="Sample rate in MHz (default: 2.0)",
    )
    parser.add_argument(
        "--device-index", type=int, default=0, help="RTL-SDR device index (default: 0)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging of messages and decoding",
    )
    parser.add_argument(
        "--gain",
        type=str,
        default="auto",
        help='SDR gain in dB or "auto" for automatic (default: auto)',
    )

    args = parser.parse_args()

    # Convert MHz to Hz
    sample_rate = args.sample_rate * 1e6

    # Create and run scanner
    scanner = ADSBScanner(sample_rate=sample_rate, debug=args.debug)

    # Handle gain setting
    if args.gain == "auto":
        scanner.gain = "auto"
    else:
        try:
            scanner.gain = float(args.gain)
        except ValueError:
            print(f"Invalid gain value: {args.gain}. Using automatic gain.")
            scanner.gain = "auto"

    if args.debug:
        print(
            "🐛 Debug mode enabled - detailed logging will be saved to 'adsb_debug.log'"
        )
        print(
            "   Watch the console and log file for detailed message decoding information"
        )

    scanner.run()


if __name__ == "__main__":
    main()
