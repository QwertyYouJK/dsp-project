# Audio Processing Project
This project was done during my summer internship at Robocore and Rhizo. It involves direction of arrival (DoA), real-time noise suppression, and acoustic echo cancellation (AEC).

## Overview

Six microphones arranged in a circular array are connected to an 8-channel sound card. The sound card is plugged into the PC as the multi-channel input. Two additional channels are used for monitoring/reference (depending on the program). The PC output drives a pair of speakers for monitoring.

Objectives:
- When recording human voice, suppress background noise
- Using the microphone array, estimate the angle a voice is coming from (DoA)
- Reduce/remove speaker playback that gets picked up by the microphones (AEC)

## Repository Structure

```
dsp-project/
├── experiments/        # components tests/prototypes
│   ├── ...
├── final/              # runnable programs
│   ├── ...
├── .gitignore
└── README.md
```

`experiments/` contains test programs for individual components.

`final/` contains the completed runnable programs.

Each directory contains its own README with build/run details.

## Installation and Setup

Developed and tested on:
- **OS**: Ubuntu 22.04 LTS
- **Compiler**: g++ (C++17)

Development Tools:
```bash
sudo apt update
sudo apt install -y build-essential pkg-config git
```

Common dependencies (varies by folder):
- PortAudio, RNNoise, FFTW3, libsndfile, WebRTC Audio Processing, pkg-config, pthread

```bash
sudo apt install -y \
  portaudio19-dev \
  librnnoise-dev \
  libfftw3-dev \
  libsndfile1-dev \
  libwebrtc-audio-processing-dev
```
See each folder README for exact build commands.

## Running the Programs

Two final programs:

- `aec_recorder`: Records 5 seconds from the microphone array (and monitor/reference if applicable). Processes audio in real time with noise suppression and echo cancellation. Intended outcome: speech remains while speaker playback is strongly reduced.

- `realtime_tdoa`: Runs until Enter/Return is pressed. Captures audio and prints the estimated angle of arrival to the terminal.

## Future Improvements

- Integrate AEC into the real-time DoA path so speaker playback doesn't corrupt TDOA/DoA
- Add beamforming to focus on a target direction and suppress off-axis talkers
