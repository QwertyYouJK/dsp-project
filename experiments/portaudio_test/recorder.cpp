// pa_record_8ch.cpp
// Build: g++ -O2 -std=c++17 pa_record_8ch.cpp -lportaudio -o pa_record_8ch
// Run:   PA_ALSA_PLUGHW=1 ./pa_record_8ch

#include <portaudio.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>

#define DEVICE_INDEX 3
#define SAMPLE_RATE 48000
#define CHANNELS 8
#define FRAMES_PER_BUFFER 256
#define RECORD_SECONDS 5

#pragma pack(push, 1)
struct WavHeader
{
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunkSize = 0;
    char wave[4] = {'W', 'A', 'V', 'E'};

    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t subchunk1Size = 16;
    uint16_t audioFormat = 1; // PCM
    uint16_t numChannels = CHANNELS;
    uint32_t sampleRate = SAMPLE_RATE;
    uint32_t byteRate = 0;
    uint16_t blockAlign = 0;
    uint16_t bitsPerSample = 16;

    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t dataSize = 0;
};
#pragma pack(pop)

static int16_t float_to_i16(float x)
{
    float v = x * 32767.0f;
    if (v > 32767.0f)
        v = 32767.0f;
    if (v < -32768.0f)
        v = -32768.0f;
    return (int16_t)lrintf(v);
}

static void die_pa(const char *where, PaError err)
{
    std::cerr << where << ": " << Pa_GetErrorText(err) << " (" << err << ")\n";
    std::exit(1);
}

int main()
{
    PaError err = Pa_Initialize();
    if (err != paNoError)
        die_pa("Pa_Initialize", err);

    std::cout << "\n\n";

    PaStreamParameters inParams{};
    inParams.device = DEVICE_INDEX;
    if (inParams.device == paNoDevice)
        die_pa("No such input device", paInvalidDevice);

    const PaDeviceInfo *di = Pa_GetDeviceInfo(inParams.device);
    inParams.channelCount = CHANNELS;
    inParams.sampleFormat = paFloat32;
    inParams.suggestedLatency = di ? di->defaultLowInputLatency : 0.0;
    inParams.hostApiSpecificStreamInfo = nullptr;

    PaStream *stream = nullptr;
    err = Pa_OpenStream(&stream, &inParams, nullptr, SAMPLE_RATE, FRAMES_PER_BUFFER, paNoFlag, nullptr, nullptr);
    if (err != paNoError)
        die_pa("Pa_OpenStream", err);

    err = Pa_StartStream(stream);
    if (err != paNoError)
        die_pa("Pa_StartStream", err);

    FILE *f = std::fopen("normal.wav", "wb");
    if (!f)
    {
        std::cerr << "Failed to open output file\n";
        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }

    WavHeader hdr;
    hdr.blockAlign = hdr.numChannels * (hdr.bitsPerSample / 8);
    hdr.byteRate = hdr.sampleRate * hdr.blockAlign;
    std::fwrite(&hdr, sizeof(hdr), 1, f);

    const int totalFrames = (int)llround((double)RECORD_SECONDS * (double)SAMPLE_RATE);

    std::vector<float> in((size_t)FRAMES_PER_BUFFER * (size_t)CHANNELS);
    std::vector<int16_t> out((size_t)FRAMES_PER_BUFFER * (size_t)CHANNELS);

    std::cout << "Recording for " << RECORD_SECONDS << " seconds...\n";

    int framesDone = 0;
    while (framesDone < totalFrames)
    {
        int framesToRead = FRAMES_PER_BUFFER;
        if (framesDone + framesToRead > totalFrames)
            framesToRead = totalFrames - framesDone;

        err = Pa_ReadStream(stream, in.data(), framesToRead);
        if (err != paNoError && err != paInputOverflowed)
            die_pa("Pa_ReadStream", err);

        const int samples = framesToRead * CHANNELS;
        for (int i = 0; i < samples; ++i)
            out[i] = float_to_i16(in[i]);

        std::fwrite(out.data(), sizeof(int16_t), (size_t)samples, f);
        hdr.dataSize += (uint32_t)((size_t)samples * sizeof(int16_t));

        framesDone += framesToRead;
    }

    hdr.chunkSize = 36 + hdr.dataSize;
    std::fseek(f, 0, SEEK_SET);
    std::fwrite(&hdr, sizeof(hdr), 1, f);
    std::fclose(f);

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    std::cout << "Saved: normal.wav\n";
    return 0;
}
