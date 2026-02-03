# Final Programs

This folder contains the two main runnable programs built from the project work:
1) a speech-focused recorder with echo/playback reduction
2) a real-time DoA estimator (TDOA/GCC-PHAT) with noise/echo handling

## Structure


```
final/
├── aec_recorder.cpp            # 5s recorder with noise suppression + playback leakage reduction
├── realtime_tdoa.cpp           # real-time DoA (TDOA/GCC-PHAT) with noise suppression
└── README.md
```

## Hardware / signal setup assumed

Hardware used:
- **Mic array:** 6 microphones arranged in a circular array
- **Interface:** 8-channel input device (USB sound card / interface)
- **Channels:** first 6 channels are mics, remaining channels are used as monitor/reference (depending on program)
- **Output:** PC audio output drives speakers/headphones used for monitoring

Important:
- Both programs assume sample rate **48 kHz**

## Programs

## 1) `aec_recorder.cpp`

### Purpose
Record a short clip (~5 seconds) of multi-channel audio and produce a cleaner output where:
- background noise is reduced
- speaker playback leakage (PC audio) is reduced as much as possible

This program is meant for making a usable recording with a clear voice even when monitoring audio is enabled.

### What it does (pipeline)
1. **Capture**: reads frames from the multi-channel input device (PortAudio).
2. **Noise suppression**: runs RNNoise on the target channel(s) to reduce steady background noise.
3. **Write to WAV**: writes output audio to a file (libsndfile).

### Inputs / outputs
- **Input:** multi-channel stream from your audio interface
- **Output:** a WAV file (see source for filename and channel format)

### Expected result
Run a basic test:
1. Start the program.
2. Play music on the PC speakers at low-to-moderate volume.
3. Speak near the array during the 5 seconds.

Expected:
- The output recording emphasizes speech.
- Background noise is reduced.
- PC playback is reduced but may not be fully removed, especially in reflective rooms or with loud playback.

### Note
- Speech may sound “processed” (normal RNNoise artifact).


## 2) `realtime_tdoa.cpp`

### Purpose
Estimate the direction of arrival (DoA) of speech in real time and print the angle to the terminal. Audio monitoring is enabled, so speaker will playback what the microphones received.

This program is meant for live direction estimation, not for producing the cleanest recording.

### What it does (pipeline)
1. **Capture**: continuously reads multi-channel frames from the mic array (PortAudio).
2. **Noise suppression (RNNoise)**: used to reduce background noise and improve reliability of event detection (and/or listening monitor).
3. **Event gating**: only runs TDOA when the sound is “worth processing” (typically based on loudness/energy).
4. **Echo-aware gating / double-talk detection**:
   - compares microphone channels to monitor/reference channels (e.g., ch7/ch8)
   - if strong correlation suggests the array is mostly hearing the speakers, TDOA is suppressed to avoid false angles
5. **TDOA (GCC-PHAT)**:
   - computes pairwise delays between selected microphone pairs using FFTW (GCC-PHAT)
6. **Angle selection**:
   - chooses the angle that best matches the measured delays given the array geometry
7. **Output**:
   - prints angle estimates to the terminal (and may optionally play monitor audio depending on the program)

### Notes on feedback (Larsen effect)
If the mic array is close to the speakers and monitoring volume is high, there can be a feedback loop.
Mitigations:
- reduce monitor volume (GAIN and LIM_THRESH variables)
- Use AEC/noise suppression

### Expected result
A basic sanity test:
1. Stand ~1–2 meters from the array at a known direction.
2. Speak short phrases.
3. Watch the printed angle.

Expected:
- When you speak from one direction, angle outputs should cluster near that direction.
- Jitter is normal. In a typical room, ±10–30° jitter is common without heavy smoothing.
- When PC playback dominates (sound through speakers), TDOA **should** reduce updates or stop updating (if echo gating is working).

### Additional Notes
- `IN_DEV` and `OUT_DEV` depends on device, use `/portaudio_test/pa_devices.cpp` to check device index
- Change constant variables `RENDER_TH`, `MIC_TH`, `NEAR_RATIO`, and `ECHO_CORR_TH` for better double-talk detection results.


## Build and Run (Linux)

Tooling:
```bash
sudo apt update
sudo apt install -y build-essential pkg-config
```

Libraries used in `final/`:
- PortAudio (capture/playback)
- RNNoise (noise suppression)
- FFTW3 (GCC-PHAT/FFT)
- libsndfile (WAV writing)
- WebRTC Audio Processing (AEC)

Install (Ubuntu 22.04):
```bash
sudo apt install -y \
  portaudio19-dev \
  librnnoise-dev \
  libfftw3-dev \
  libsndfile1-dev \
  libwebrtc-audio-processing-dev
```

Build `aec_recorder`:
```bash
g++ -O3 -std=c++17 aec_recorder.cpp -o aec_recorder \
  $(pkg-config --cflags --libs portaudio-2.0 sndfile rnnoise) \
  -lm -pthread
```

Build `realtime_tdoa`:
```bash
g++ -O3 -std=c++17 realtime_tdoa.cpp -o realtime_tdoa \
  $(pkg-config --cflags --libs portaudio-2.0 rnnoise fftw3 webrtc-audio-processing-2) \
  -lm -pthread
```

Run `aec_recorder`:
```bash
./aec_recorder
```

Run `realtime_tdoa`:
```bash
./realtime_tdoa
```

### ALSA plughw note
If your device cannot do 48 kHz natively, you must run with ALSA `plughw` conversion:
```bash
PA_ALSA_PLUGHW=1 ./aec_recorder
PA_ALSA_PLUGHW=1 ./realtime_tdoa
```

## Expected Results
- `aec_recorder`: output WAV emphasizes speech; background noise reduced; speaker playback leakage reduced but not guaranteed to be zero.
- `realtime_tdoa`: prints angle estimates that cluster near the real speaker location; reduces/halts updates when playback dominates (if echo-aware gating is enabled).