# Acoustic Echo Cancellation (AEC)

This folder contains test programs and prototypes for removing speaker playback (echo/feedback) from microphone recordings.

## Structure

```
aec_test/
├── music_remove.cpp    # AEC using a reference/monitor channel (ch7/ch8)
├── speex_test.cpp      # prototype AEC with SpeexDSP
├── webrtc_test.cpp     # prototype AEC with WebRTC Audio Processing
└── README.md
```


## What “AEC” means in this project

AEC removes **playback sound from the speakers** that leaks back into the microphones.

Typical sources of echo here:
- Music/audio playing on the PC while recording
- Monitoring enabled (mic routed to speakers/headphones), then re-captured by the mic array
- General acoustic coupling between speakers and the mic array in the same room

Target outcome:
- Speech stays intelligible
- Speaker playback is strongly reduced in the recorded mic signal

## Signal routing used by these prototypes

Most AEC implementations need two signals:
- **Mic (near-end):** what the microphone array captures
- **Reference (far-end):** what is being sent to the speaker (the AEC “knows” this and subtracts its contribution)

The reference is can be taken from a **monitor/reference channel** (often channel 7/8 of the 8-channel interface) or from the same audio stream being played to the output device, depending on the program.

Exact channel mapping and assumptions are documented in each `.cpp` file.

## Files

### `speex_test.cpp`
Early prototype using **SpeexDSP** echo canceller.
- Purpose: validate the basic pipeline and signal routing.
- Notes: worked as a baseline but cancellation quality was limited for this setup.

### `webrtc_test.cpp`
Prototype using **WebRTC Audio Processing** AEC.
- Purpose: better cancellation and robustness compared to SpeexDSP.
- Notes: WebRTC AEC is sensitive to correct sample rate, frame size, and clean reference routing.

### `music_remove.cpp`
AEC experiment focused on removing a known playback signal (music) using a dedicated reference channel.
- Purpose: stress-test reference alignment and cancellation strength.
- Notes: best used when the reference channel truly carries the playback signal with minimal distortion.

## Build notes (Linux)

These prototypes are usually built with `g++` and `pkg-config`.

Tooling:
```bash
sudo apt update
sudo apt install -y build-essential pkg-config
```

### WebRTC version
Install:
```bash
sudo apt install portaudio19-dev libwebrtc-audio-processing-dev
```
Build:
```bash
g++ -O3 -std=c++17 webrtc_test.cpp -o webrtc_test \
  $(pkg-config --cflags --libs webrtc-audio-processing-2 portaudio-2.0) \
  -lm -pthread
```
Run:
```bash
./webrtc_test
```

### SpeexDSP version
Install:
```bash
sudo apt install portaudio19-dev libspeexdsp-dev
```
Build:
```bash
g++ -O3 -std=c++17 speex_test.cpp -o speex_test \
  $(pkg-config --cflags --libs speexdsp portaudio-2.0) \
  -lm -pthread
```
Run:
```bash
./speex_test
```

### Reference-channel removal test
Install:
```bash
sudo apt install portaudio19-dev libwebrtc-audio-processing-dev libsndfile1-dev
```
Build:
```bash
g++ -O3 -std=c++17 music_remove.cpp -o music_remove \
  $(pkg-config --cflags --libs webrtc-audio-processing-2 portaudio-2.0 sndfile) \
  -lm -pthread
```
Run:
```bash
./music_remove
```

### ALSA plughw note (troubleshooting)
If your device cannot do 48 kHz natively and you get error message: `Pa_OpenStream(in): Invalid sample rate (-9997)`, you must run with ALSA `plughw` conversion:
```bash
PA_ALSA_PLUGHW=1 ./<program_file>
```

## Expected Results
- Playback level drops significantly in the processed mic output
- Speech remains mostly intelligible (some artifacts are acceptable)
- Echo/feedback stops when placing microphone near speaker