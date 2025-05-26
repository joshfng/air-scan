# ADS-B Aircraft Tracking - 1090 MHz

This document describes the ADS-B (Automatic Dependent Surveillance-Broadcast)
aircraft tracking functionality added to the SDR Scanner suite.

## Overview

ADS-B is a surveillance technology used by aircraft to broadcast their position,
velocity, and other flight information. Operating at 1090 MHz, ADS-B uses Mode S
transponder technology to transmit data that can be received by ground stations
and other aircraft.

## Features

### Core Functionality

- **Real-time aircraft tracking** at 1090 MHz
- **Mode S transponder decoding** with CRC-24 verification
- **Aircraft identification** including ICAO addresses and callsigns
- **Position tracking** with latitude/longitude coordinates
- **Flight data extraction** including:
  - Altitude (barometric and GNSS)
  - Ground speed and heading
  - Vertical rate (climb/descent)
  - Aircraft category and identification

### User Interface

- **Live map display** showing aircraft positions in real-time
- **Statistics panel** with message counts and decoding success rates
- **Aircraft database** with automatic cleanup of stale entries
- **Interactive controls** for start/stop/clear operations
- **Tabular output** with comprehensive aircraft information

### Technical Features

- **CRC-24 verification** ensures message integrity
- **Preamble detection** for reliable signal acquisition
- **Multiple message types** supported:
  - Type 1-4: Aircraft identification and category
  - Type 9-18: Airborne position (barometric altitude)
  - Type 19: Airborne velocity
  - Type 20-22: Airborne position (GNSS height)
- **Automatic gain control** optimized for aircraft signals
- **Signal processing** with correlation-based detection

## Usage

### Quick Start

```bash
# Easy way: Use the convenience script
./run_adsb_scan.sh

# Manual way: Run directly
poetry run python adsb_scanner.py

# Custom sample rate
poetry run python adsb_scanner.py --sample-rate 2.0
```

### Command Line Options

```bash
python adsb_scanner.py [options]

Options:
  --sample-rate RATE    Sample rate in MHz (default: 2.0)
  --device-index INDEX  RTL-SDR device index (default: 0)
  -h, --help           Show help message
```

### Controls

- **Start Button**: Begin ADS-B tracking
- **Stop Button**: Pause tracking (preserves aircraft database)
- **Clear Button**: Reset aircraft database and statistics

## Understanding the Output

### Aircraft Information Display

Each tracked aircraft shows:

- **ICAO Address**: Unique 24-bit aircraft identifier (e.g., "40621D")
- **Callsign**: Flight number or aircraft registration (e.g., "UAL123")
- **Position**: Latitude and longitude coordinates
- **Altitude**: Height in feet (barometric or GNSS)
- **Speed**: Ground speed in knots
- **Heading**: Direction of travel in degrees
- **Messages**: Number of messages received from this aircraft
- **Age**: Time since last message received

### Statistics Panel

- **Total Messages**: All preambles detected
- **Valid Messages**: Messages that passed CRC verification
- **Success Rate**: Percentage of valid messages
- **Total Aircraft**: All aircraft ever detected
- **Active Aircraft**: Aircraft seen in the last 60 seconds

### Example Output

```shell
TRACKED AIRCRAFT
====================================================================================================
ICAO     Callsign   Lat        Lon         Alt(ft)  Spd(kt)  Hdg    Msgs   Age(s)
----------------------------------------------------------------------------------------------------
A12345   UAL123     37.7749    -122.4194   35000    450      270°   45     12
B67890   DAL456     40.7128    -74.0060    28000    380      90°    23     8
C11111   SWA789     34.0522    -118.2437   32000    420      180°   67     5
```

## Technical Details

### ADS-B Message Structure

ADS-B messages use the Mode S format:

- **Preamble**: 8 μs synchronization pattern
- **Data**: 56 or 112 bits of information
- **CRC**: 24-bit cyclic redundancy check

### Message Types Decoded

1. **Aircraft Identification (Types 1-4)**

   - Callsign/flight number
   - Aircraft category

2. **Airborne Position (Types 9-18)**

   - Latitude and longitude (CPR encoded)
   - Altitude (barometric or GNSS)

3. **Airborne Velocity (Type 19)**

   - Ground speed and heading
   - Vertical rate

### Signal Processing

1. **Preamble Detection**: Correlation-based detection of Mode S preambles
2. **Bit Extraction**: 1 MHz Manchester encoding recovery
3. **CRC Verification**: 24-bit polynomial check for message integrity
4. **Data Decoding**: Field extraction based on message type

### Position Decoding

ADS-B uses Compact Position Reporting (CPR) encoding:

- **Local Decoding**: Requires reference position (simplified implementation)
- **Global Decoding**: Requires two messages with different formats
- Current implementation uses simplified local decoding for demonstration

## Hardware Requirements

### Antenna Recommendations

For optimal ADS-B reception:

- **Frequency**: 1090 MHz
- **Antenna Types**:
  - 1/4 wave vertical (6.8 cm)
  - Collinear array for higher gain
  - Commercial ADS-B antennas
- **Placement**: High and clear of obstructions
- **Coax**: Low-loss cable (RG-58 or better)

### SDR Settings

- **Sample Rate**: 2.0 MHz (recommended)
- **Gain**: 40 dB (high gain for weak aircraft signals)
- **Frequency**: 1090 MHz (fixed)

## Range and Coverage

### Typical Reception Range

- **Line of Sight**: Primary limitation
- **Aircraft at 35,000 ft**: ~250 nautical miles
- **Aircraft at 10,000 ft**: ~120 nautical miles
- **Ground level**: ~5-10 nautical miles

### Factors Affecting Range

- **Antenna height and gain**
- **Local terrain and obstructions**
- **Aircraft altitude and transmitter power**
- **Atmospheric conditions**
- **Interference from other sources**

## Integration with Other Tools

### Wideband Scanner

ADS-B is included as a frequency band in the wideband scanner:

```bash
# Include ADS-B in wideband scan
poetry run python wideband_scanner.py --bands ADS_B L_BAND

# ADS-B only
poetry run python wideband_scanner.py --bands ADS_B
```

### Data Export

Aircraft tracking data can be extended to export formats:

- JSON for further processing
- CSV for spreadsheet analysis
- KML for Google Earth visualization

## Troubleshooting

### No Aircraft Detected

1. **Check antenna connection and placement**
2. **Verify 1090 MHz reception with spectrum analyzer**
3. **Increase gain if signals are weak**
4. **Check for local interference sources**

### Poor Decoding Rate

1. **Improve antenna system**
2. **Reduce sample rate if CPU limited**
3. **Check for USB bandwidth issues**
4. **Move away from interference sources**

### Position Accuracy

1. **CPR decoding limitations** in simplified implementation
2. **Consider upgrading to full CPR decoder**
3. **Multiple message correlation** improves accuracy

## Legal Considerations

### Permitted Activities

- **Receiving ADS-B signals** is legal in most countries
- **Educational and hobbyist use** is generally allowed
- **Aviation safety applications** may require certification

### Restrictions

- **Do not interfere** with aviation systems
- **Respect privacy** of flight information
- **Follow local regulations** regarding radio monitoring
- **Commercial use** may require licensing

## Future Enhancements

### Potential Improvements

1. **Full CPR Position Decoding**

   - Global position resolution
   - Improved accuracy
   - Multi-message correlation

2. **Extended Message Types**

   - Surface position reports
   - Aircraft status messages
   - Emergency codes

3. **Database Integration**

   - Aircraft registration lookup
   - Flight plan correlation
   - Historical tracking

4. **Advanced Visualization**

   - 3D flight paths
   - Real-time mapping
   - Flight prediction

5. **Network Integration**

   - BEAST protocol output
   - FlightRadar24 feeding
   - Multi-receiver correlation

## References

- **ICAO Annex 10**: Standards for ADS-B
- **RTCA DO-260B**: ADS-B technical specifications
- **Mode S specification**: ICAO Doc 9871
- **RTL-SDR documentation**: Hardware specifications

## Testing

Run the test suite to verify functionality:

```bash
poetry run python test_adsb.py
```

This tests all components without requiring hardware and provides example output.

---

**Note**: This implementation provides a solid foundation for ADS-B reception and
decoding. For production aviation applications, consider using certified
equipment and software.
