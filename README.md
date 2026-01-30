# Audio Project
This project is done during my summer internship working at Robocore. It involves locating angle of a sound source (DoA), real-time noise suppression, and acoustic echo cancellation.

## Setup and Objectives

A six-microphone arranged in a circular array is attached to a sound card with 8 channels. The sound card is plugged into the computer as input. The remaining two channels are connected to two speakers for sound monitoring. The same speakers are also plugged into the PC as mono output.

The objectives:
- Perform noise suppression to only detect human voice
- Use the array of microphones to hear the voice and calculate the angle it's coming from
- Perform echo cancellation when the speakers outputs the sound back to the microphone
- Implement everything above into a pipeline that performs in real time


## What this repo is

This repo contains a collection of audio experiments written in C++ as well as a final program for showcase.

- In the `experiments/` directory, it contains test files for each module of the pipeline.
- In the `final/` directory, it contains the complete pipeline which can run in real time.

