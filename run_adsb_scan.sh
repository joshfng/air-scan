#!/usr/bin/env bash

# ADS-B Aircraft Tracker - 1090 MHz Mode S transponder decoder
# Tracks aircraft using ADS-B signals

echo "Starting ADS-B Aircraft Tracker..."
echo "Frequency: 1090 MHz (ADS-B/Mode S)"
echo "This will track aircraft transponder signals and display positions"
echo ""
echo "Usage: $0 [--debug|-d]"
echo "  --debug, -d    Enable detailed debug logging"
echo ""

# Check if SDR device is connected (cross-platform)
if command -v lsusb >/dev/null 2>&1; then
    # Linux
    if ! lsusb | grep -q "RTL"; then
        echo "Warning: No RTL-SDR device detected"
        echo "Please connect an RTL-SDR device and try again"
        echo ""
    fi
elif command -v system_profiler >/dev/null 2>&1; then
    # macOS
    if ! system_profiler SPUSBDataType 2>/dev/null | grep -q "RTL"; then
        echo "Warning: No RTL-SDR device detected"
        echo "Please connect an RTL-SDR device and try again"
        echo ""
    fi
else
    echo "Note: Cannot detect SDR device automatically on this system"
    echo ""
fi

# Set up library path for macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
    echo "Set macOS library path for RTL-SDR"
fi

# Check for debug flag
DEBUG_FLAG=""
if [[ "$1" == "--debug" || "$1" == "-d" ]]; then
    DEBUG_FLAG="--debug"
    echo "Debug mode enabled - detailed logging will be saved to 'adsb_debug.log'"
fi

# Run the ADS-B scanner
DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH" poetry run python adsb_scanner.py \
    --sample-rate 2.0 \
    --device-index 0 \
    $DEBUG_FLAG

echo "ADS-B tracking stopped"
