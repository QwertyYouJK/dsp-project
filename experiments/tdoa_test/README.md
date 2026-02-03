# Direction of Arrival (DoA) — TDOA + GCC-PHAT

This folder contains prototypes and programs that estimate direction of arrival (DoA) using time difference of arrival (TDOA) computed via GCC-PHAT.

## Structure
```
tdoa_test/
├── tdoa_test.m                 # MATLAB prototype
├── tdoa_test.cpp               # offline / scripted test in C++
├── tdoa_realtime.cpp           # real-time TDOA/DoA estimate
└── README.md
```

## High-level Approach

To estimate the angle of a sound source, this project measures TDOA between microphone pairs.

Core idea:
- A sound reaches different microphones at slightly different times.
- Estimate those delays (in samples or seconds), and the arrival direction can be infered based on the microphone geometry.

### GCC-PHAT
For two microphone signals `x1[n]` and `x2[n]`, GCC-PHAT estimates the relative delay by:
1. FFT both signals
2. Compute cross-spectrum `X1 * conj(X2)`
3. Apply PHAT weighting (normalize magnitude) to emphasize timing over tone
4. IFFT back to a correlation-like function
5. Pick the peak within a valid lag range → estimated delay

Then multiple pairwise delays are combined to choose the angle that best matches the whole array.

### Note:
- This works best with broadband transients (claps, consonants) and reasonable SNR.
- Reverberation, multiple talkers, and speaker playback can produce incorrect peaks.

## Files

### `tdoa_test.m`
MATLAB prototype used for proof-of-concept and as a reference for the C++ port.

Typical usage:
- Load a recording
- Select a short time window around a known event (e.g., a clap)
- Compute pairwise TDOA and estimate angle

### `tdoa_test.cpp`
C++ offline prototype.
- Uses a hard-coded timestamp to cut a segment from audio
- Runs GCC-PHAT/TDOA on that segment
- Prints the estimated DoA

This is mainly for debugging correctness against MATLAB.

### `tdoa_realtime.cpp`
C++ real-time prototype.
- Opens the microphone array stream continuously
- Detects loud moments
- Runs GCC-PHAT/TDOA periodically and prints the angle to the terminal

## Build notes (Linux)

Tooling:
```bash
sudo apt update
sudo apt install -y build-essential pkg-config
```

Install dependencies:
```bash
sudo apt install -y portaudio19-dev libfftw3-dev
```

Build (real-time):
```bash
g++ -O3 -std=c++17 tdoa_realtime.cpp -o tdoa_realtime \
  $(pkg-config --cflags --libs portaudio-2.0 fftw3) \
  -lm -pthread
```

Build (offline/prototype):
```bash
g++ -O3 -std=c++17 tdoa_test.cpp -o tdoa_test \
  $(pkg-config --cflags --libs fftw3) \
  -lm -pthread
```

Run:
```bash
./tdoa_realtime
./tdoa_test
```

### ALSA plughw note
If your device cannot do 48 kHz natively, you must run with ALSA `plughw` conversion:
```bash
PA_ALSA_PLUGHW=1 ./<program_file>
```

### Expected Result
- The printed estimated angle should be near the direction of the sound (within ±10–30°)
