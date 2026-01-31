# Block diagram

[Electrodes] --> [Input protection] --> [AFE (diff amp + ADC)] --> [MCU]
                                               |
                                               +--> [Bias/DRL] --> [Bias electrode]

MCU --> (BLE) --> Phone/PC --> DSP --> ML model --> decoded token/text --> (TTS) --> audio/bone conduction
