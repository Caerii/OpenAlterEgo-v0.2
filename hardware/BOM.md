# BOM (starter)

This is *not* the final BOM — it’s a menu.

## V0 Benchtop
- Biopotential AFE dev board (ADS1299-based) **or** OpenBCI Cyton/Ganglion
- Pre-gelled Ag/AgCl electrodes (8–10)
- Electrode lead wires / snap adapters
- Battery pack (LiPo / power bank)
- Optional: USB isolator (for safety)

## V1 Wearable PCB (conceptual)
- AFE: ADS1299 (8ch, 24-bit) or similar biopotential AFE
- MCU+BLE: nRF52840 (or similar)
- LiPo charger IC (e.g., MCP73831-class)
- LDO regulators (separate analog/digital rails)
- ESD diodes + series resistors on electrode inputs
- Connectors for electrodes (snap / JST / pogo)
- Optional: 9-axis IMU

## Output
- Bone conduction transducer / headphones (or just a normal earbud for dev)

