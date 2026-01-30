#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <portaudio.h>

#define SAMPLE_RATE 48000
#define CHANNELS 8
#define FRAMES_PER_BUFFER 256
#define RECORD_SECONDS 5

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

static int16_t float_to_i16(float x) {
    if (x > 1.0f) x = 1.0f;
    if (x < -1.0f) x = -1.0f;
    return (int16_t)std::lrintf(x * 32767.0f);
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

    FILE* f = std::fopen("recording_8ch.wav", "wb");
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

    std::vector<float> in(FRAMES_PER_BUFFER * CHANNELS);
    std::vector<int16_t> out(FRAMES_PER_BUFFER * CHANNELS);

    int totalFrames = RECORD_SECONDS * SAMPLE_RATE;
    int framesWritten = 0;

    std::cout << "Recording " << RECORD_SECONDS << " seconds...\n";

    while (framesWritten < totalFrames) {
        int framesToRead = FRAMES_PER_BUFFER;
        if (framesWritten + framesToRead > totalFrames)
            framesToRead = totalFrames - framesWritten;

        err = Pa_ReadStream(stream, in.data(), framesToRead);
        if (err != paNoError) break;

        for (int i = 0; i < framesToRead * CHANNELS; i++) {
            out[i] = float_to_i16(in[i]);
        }

        std::fwrite(out.data(), sizeof(int16_t), framesToRead * CHANNELS, f);
        hdr.dataSize += framesToRead * CHANNELS * sizeof(int16_t);
        framesWritten += framesToRead;
    }

    // Patch header
    hdr.chunkSize = 36 + hdr.dataSize;
    std::fseek(f, 0, SEEK_SET);
    std::fwrite(&hdr, sizeof(hdr), 1, f);

    std::fclose(f);

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    std::cout << "Saved: recording_8ch.wav\n";
    return 0;
}