#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdint>
#include <portaudio.h>
#include "rnnoise.h"

#define SAMPLE_RATE 48000
#define CHANNELS 8
#define FRAMES_PER_BUFFER 256
#define RECORD_SECONDS 5

constexpr int RN_FRAME = 480;

// ---------- WAV header (PCM 16-bit) ----------
#pragma pack(push, 1)
struct WavHeader {
    char riff[4] = {'R','I','F','F'};
    uint32_t chunkSize;
    char wave[4] = {'W','A','V','E'};

    char fmt[4] = {'f','m','t',' '};
    uint32_t subchunk1Size = 16;
    uint16_t audioFormat = 1; // PCM
    uint16_t numChannels = CHANNELS;
    uint32_t sampleRate = SAMPLE_RATE;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample = 16;

    char data[4] = {'d','a','t','a'};
    uint32_t dataSize;
};
#pragma pack(pop)

static int16_t rnfloat_to_i16(float x) {
    if (x > 32767.0f) x = 32767.0f;
    if (x < -32768.0f) x = -32768.0f;
    return (int16_t)std::lrintf(x);
}

int main() {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio init error\n";
        return 1;
    }

    PaStream* stream;
    err = Pa_OpenDefaultStream(
        &stream,
        CHANNELS,
        0,
        paFloat32,
        SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        nullptr,
        nullptr
    );

    if (err != paNoError) {
        std::cerr << "Failed to open stream\n";
        Pa_Terminate();
        return 1;
    }

    Pa_StartStream(stream);

    FILE* f = std::fopen("recording_8ch_nc.wav", "wb");
    if (!f) {
        std::cerr << "Failed to open output file\n";
        return 1;
    }

    // Write placeholder header
    WavHeader hdr;
    hdr.blockAlign = hdr.numChannels * (hdr.bitsPerSample / 8);
    hdr.byteRate = hdr.sampleRate * hdr.blockAlign;
    hdr.dataSize = 0;
    hdr.chunkSize = 36 + hdr.dataSize;
    std::fwrite(&hdr, sizeof(hdr), 1, f);

    // ---- RNNoise states: one per channel ----
    std::vector<DenoiseState*> st(CHANNELS, nullptr);
    for (int ch = 0; ch < CHANNELS; ch++) {
        st[ch] = rnnoise_create(nullptr);
        if (!st[ch]) {
            std::cerr << "rnnoise_create failed\n";
            for (int k = 0; k < ch; k++) rnnoise_destroy(st[k]);
            std::fclose(f);
            Pa_StopStream(stream);
            Pa_CloseStream(stream);
            Pa_Terminate();
            return 1;
        }
    }

    std::vector<float> in(FRAMES_PER_BUFFER * CHANNELS);
    
    // Per-channel pending buffers (so we can assemble 480-sample frames)
    std::vector<std::vector<float>> pending(CHANNELS);
    for (auto &p : pending) p.reserve(RN_FRAME * 2);

    // RNNoise frame buffers
    std::vector<float> frameIn(RN_FRAME), frameOut(RN_FRAME);

    // Output (interleaved int16) for one RN frame across all channels
    std::vector<int16_t> outInterleaved(RN_FRAME * CHANNELS);

    int totalFrames = RECORD_SECONDS * SAMPLE_RATE;
    int framesWritten = 0;

    std::cout << "Recording " << RECORD_SECONDS << " seconds...\n";

    while (framesWritten < totalFrames) {
        int framesToRead = FRAMES_PER_BUFFER;
        if (framesWritten + framesToRead > totalFrames)
            framesToRead = totalFrames - framesWritten;

        err = Pa_ReadStream(stream, in.data(), framesToRead);
        if (err != paNoError) break;

        // De-interleave and append into pending buffers
        for (int fidx = 0; fidx < framesToRead; fidx++) {
            for (int ch = 0; ch < CHANNELS; ch++) {
                pending[ch].push_back(in[fidx * CHANNELS + ch]);
            }
        }

        framesWritten += framesToRead;

        // Process in lockstep: whenever we have >=480 samples, denoise all channels and write 480 frames.
        while ((int)pending[0].size() >= RN_FRAME) {
            for (int ch = 0; ch < CHANNELS; ch++) {

                // copy & SCALE to RNNoise expected range
                for (int n = 0; n < RN_FRAME; n++) {
                    frameIn[n] = pending[ch][n] * 32768.0f;
                }

                rnnoise_process_frame(st[ch], frameOut.data(), frameIn.data());

                // write interleaved output (frameOut is also int16-range)
                for (int n = 0; n < RN_FRAME; n++) {
                    outInterleaved[n * CHANNELS + ch] = rnfloat_to_i16(frameOut[n]);
                }

                pending[ch].erase(pending[ch].begin(), pending[ch].begin() + RN_FRAME);
            }

            std::fwrite(outInterleaved.data(), sizeof(int16_t), outInterleaved.size(), f);
            hdr.dataSize += (uint32_t)(outInterleaved.size() * sizeof(int16_t));
        }
    }

    // Patch header
    hdr.chunkSize = 36 + hdr.dataSize;
    std::fseek(f, 0, SEEK_SET);
    std::fwrite(&hdr, sizeof(hdr), 1, f);

    std::fclose(f);

    for (int ch = 0; ch < CHANNELS; ch++) {
        rnnoise_destroy(st[ch]);
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    std::cout << "Saved: recording_8ch_nc.wav\n";
    return 0;
}