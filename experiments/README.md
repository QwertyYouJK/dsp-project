# Experiments

This directory contains test programs and prototypes built during the project.

## Structure

```
experiments/
├── aec_test/                   # test protoypes on acoustic echo cancellation
│   ├── ...
├── ns_test/                    # prototype on noise suppression (RNNoise)
│   ├── ...
├── portaudio_test/             # test files on PortAudio
│   ├── ...
├── tdoa_test/                  # TDOA implementation prototypes
│   ├── ...
├── mic coords.png              # diagram on mic positions
├── mic_coords + plan.txt       # notes on mic pair distances and GCC-PHAT
└── README.md
```

## What’s inside
- `aec_test/`: AEC experiments and test harnesses
- `ns_test/`: RNNoise integration tests
- `portaudio_test/`: PortAudio smoke tests (device listing, stream configs, capture/playback)
- `tdoa_test/`: GCC-PHAT/TDOA/DoA experiments

Each directory contains its own README with exact build/run details and dependencies.

## Recommended entry points
If you only want the “best” version of each component, start with:
- AEC: `aec_test/webrtc_test.cpp`
- TDOA/DoA: `tdoa_test/tdoa_realtime_lnx.cpp`
- RNNoise: `ns_test/rnnoise.cpp`
