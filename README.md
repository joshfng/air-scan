# ADS-B Aircraft Tracker

[![CI](https://github.com/joshfng/air-scan/workflows/CI/badge.svg)](https://github.com/USERNAME/REPOSITORY/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Real-time aircraft tracking using 1090 MHz Mode S transponder signals with
RTL-SDR devices.

## Overview

This project provides a comprehensive ADS-B (Automatic Dependent
Surveillance-Broadcast) aircraft tracking system that:

- **Tracks aircraft in real-time** using 1090 MHz Mode S transponder signals
- **Displays aircraft positions** on an interactive map with live updates
- **Shows aircraft details** including callsign, altitude, speed, heading, and
  vertical rate
- **Provides signal analysis tools** for debugging and optimization
- **Supports automatic gain control** for optimal signal reception

## Features

### Core Functionality

- **Real-time aircraft tracking** at 1090 MHz
- **Mode S transponder decoding** with CRC-24 verification
- **Aircraft identification** (ICAO addresses, callsigns)
- **Position tracking** (latitude/longitude, altitude)
- **Flight data** (speed, heading, vertical rate)
- **Live map display** with aircraft positions
- **Statistics panel** with tracking information

### Signal Processing

- **Advanced preamble detection** using signal correlation
- **Dynamic thresholding** based on signal percentiles
- **Noise reduction** with moving average filtering
- **Automatic gain control** for optimal signal resolution
- **Multiple message type support** (DF=17/18 ADS-B Extended Squitter)

### Debug and Analysis Tools

- **Comprehensive debug logging** to console and file
- **Signal strength analysis** and threshold optimization
- **Message decoding statistics** with success rates
- **Signal analyzer** for antenna and gain optimization
- **Bit-level debugging** for troubleshooting reception issues

## Hardware Requirements

### RTL-SDR Device

- **RTL-SDR USB dongle** (RTL2832U + R820T/R820T2/R828D tuner)
- **Frequency range** covering 1090 MHz
- **Recommended models**: RTL-SDR Blog V3/V4, NooElec NESDR

### Antenna

- **1090 MHz antenna** (quarter-wave ~6.8cm or specialized ADS-B antenna)
- **Good line of sight** for aircraft reception
- **Proper grounding** and impedance matching recommended

### System Requirements

- **Python 3.10+**
- **USB 2.0/3.0 port** for RTL-SDR device
- **4GB+ RAM** recommended for real-time processing
- **Linux/macOS/Windows** (tested on macOS)

## Installation

### Prerequisites

```bash
# Install RTL-SDR drivers (macOS with Homebrew)
brew install librtlsdr

# Or on Ubuntu/Debian
sudo apt-get install rtl-sdr librtlsdr-dev

# Or on Windows - use Zadig to install WinUSB drivers
```

### Python Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd adsb-aircraft-tracker

# Install with Poetry (recommended)
poetry install
poetry shell

# Or with pip
pip install pyrtlsdr numpy matplotlib scipy
```

## Usage

### Basic Aircraft Tracking

```bash
# Start the ADS-B aircraft tracker
poetry run python adsb_scanner.py

# Or with debug logging
poetry run python adsb_scanner.py --debug

# With custom gain setting
poetry run python adsb_scanner.py --gain 30.0

# Using convenience script
./run_adsb_scan.sh
```

### Signal Analysis Tools

#### Signal Analyzer (Recommended First Step)

```bash
# Analyze signal quality and find optimal settings
poetry run python adsb_signal_analyzer.py
```

This tool will:

- Test antenna connection
- Find optimal gain settings
- Analyze spectrum around 1090 MHz
- Detect ADS-B activity
- Provide specific recommendations

#### Threshold Optimization

```bash
# Find optimal signal threshold
poetry run python find_optimal_threshold.py
```

#### Simple Signal Test

```bash
# Quick signal quality test
poetry run python test_adsb_simple.py
```

#### Debug Tools

```bash
# Focused signal debugging
poetry run python debug_adsb_signals.py

# Bit inversion testing
poetry run python test_bit_inversion.py
```

### Command Line Options

```bash
python adsb_scanner.py [options]

Options:
  --sample-rate RATE    Sample rate in MHz (default: 2.0)
  --device-index INDEX  RTL-SDR device index (default: 0)
  --gain GAIN          SDR gain in dB or "auto" (default: auto)
  --debug              Enable detailed debug logging
  --help               Show help message
```

## Interface

### Real-time Map Display

- **Aircraft positions** plotted on interactive map
- **Aircraft labels** showing callsign/ICAO and altitude
- **Live updates** as aircraft move
- **Zoom and pan** capabilities

### Statistics Panel

- **Total/valid message counts** and success rates
- **Active aircraft count** and details
- **Recent aircraft list** with last seen times
- **Signal strength indicators**

### Control Buttons

- **Start/Stop** scanning
- **Clear** aircraft database
- **Real-time updates** every 2 seconds

## Debug Logging

When `--debug` is enabled, detailed information is logged to:

- **Console output** for real-time monitoring
- **adsb_debug.log** file for detailed analysis

Debug information includes:

- Signal strength and threshold analysis
- Preamble detection details
- Message decoding attempts
- CRC validation results
- Aircraft tracking updates

## Troubleshooting

### No Aircraft Detected

1. **Check signal analyzer output** - run `adsb_signal_analyzer.py`
2. **Verify antenna connection** - should see power increase with gain
3. **Check for ADS-B activity** - use flight tracking websites
4. **Try different gain settings** - automatic gain usually works best
5. **Improve antenna placement** - higher elevation, clear line of sight

### Poor Signal Quality

1. **Use automatic gain control** - `--gain auto` (default)
2. **Check antenna connection** and impedance matching
3. **Reduce interference** - move away from electronics
4. **Improve antenna** - use dedicated 1090 MHz ADS-B antenna

### CRC Errors

1. **Check signal strength** - should see strong signals in analyzer
2. **Verify threshold settings** - use `find_optimal_threshold.py`
3. **Check for quantization** - automatic gain helps
4. **Antenna issues** - poor connection can cause bit errors

## Technical Details

### ADS-B Protocol

- **Frequency**: 1090 MHz
- **Modulation**: Pulse Position Modulation (PPM)
- **Message types**: DF=17/18 Extended Squitter
- **Data rate**: 1 Mbps
- **Message length**: 112 bits (short) or 224 bits (long)

### Signal Processing Implementation

- **Sample rate**: 2 MHz (configurable)
- **Preamble detection**: Mode S pattern correlation
- **Bit extraction**: Multi-point sampling with maximum detection
- **Threshold**: 99th percentile adaptive thresholding
- **CRC validation**: CRC-24 polynomial verification

### Message Decoding

- **Aircraft identification**: Callsign extraction from Type 1-4 messages
- **Position decoding**: CPR (Compact Position Reporting) format
- **Velocity decoding**: Ground speed and heading calculation
- **Altitude decoding**: Barometric and GNSS altitude

## Files

### Core Components

- `adsb_scanner.py` - Main ADS-B aircraft tracker with GUI
- `run_adsb_scan.sh` - Convenience script for easy startup

### Analysis Tools

- `adsb_signal_analyzer.py` - Comprehensive signal analysis
- `find_optimal_threshold.py` - Signal threshold optimization
- `debug_adsb_signals.py` - Focused debugging tool
- `test_adsb_simple.py` - Simple signal quality test
- `test_bit_inversion.py` - Bit inversion testing

### Testing

- `test_adsb.py` - Comprehensive test suite

### Debug

- `adsb_debug.log` - Debug output log (generated during operation)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)
file for details.

## Acknowledgments

- RTL-SDR community for hardware support
- ADS-B protocol specifications
- Open source SDR software ecosystem
