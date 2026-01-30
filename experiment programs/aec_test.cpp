// aec_test.cpp
// Run:   PA_ALSA_PLUGHW=1 ./aec_test

#include <iostream>
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>
#include <portaudio.h>
#include <speex/speex_echo.h>
#include <speex/speex_preprocess.h>
#include <cstdint>

constexpr int IN_DEV = 3;
constexpr int OUT_DEV = 4;

constexpr int IN_CH = 8;
constexpr int OUT_CH = 1;

constexpr int SAMPLE_RATE = 48000;
constexpr int FRAME = 480;

constexpr float GAIN = 0.02f;

static void die_pa(const char *where, PaError err)
{
    std::cerr << where << ": " << Pa_GetErrorText(err) << " (" << err << ")\n";
    std::exit(1);
}

double rms_ch(int ch, const std::vector<float> &in)
{
    double s = 0.0;
    for (int i = 0; i < FRAME; i++)
    {
        float v = in[(size_t)i * (size_t)IN_CH + (size_t)ch];
        s += (double)v * (double)v;
    }
    return std::sqrt(s / FRAME);
}

// helpers
static inline spx_int16_t f32_to_i16(float x)
{
    if (x > 1.0f)
        x = 1.0f;
    if (x < -1.0f)
        x = -1.0f;
    return (spx_int16_t)lrintf(x * 32767.0f);
}
static inline float i16_to_f32(spx_int16_t x)
{
    return (float)x / 32768.0f;
}

std::vector<spx_int16_t> play_to_spx(const std::vector<float> &play)
{
    std::vector<spx_int16_t> result((size_t)FRAME);
    for (int i = 0; i < FRAME; i++)
    {
        result[i] = f32_to_i16(play[i]);
    }
    return result;
}

std::vector<float> inline spx_to_play(const std::vector<spx_int16_t> &src)
{
    std::vector<float> result((size_t)FRAME);
    for (int i = 0; i < FRAME; i++)
    {
        result[i] = i16_to_f32(src[i]);
    }
    return result;
}

int main()
{
    PaError err = Pa_Initialize();
    if (err != paNoError)
        die_pa("Pa_Initialize", err);

    std::atomic<bool> run{true};
    std::thread stopper([&]
                        {
        std::string line;
        std::getline(std::cin, line);
        run = false; });

    // ---------- Input stream ----------
    PaStream *inStream = nullptr;
    PaStreamParameters inParams{};
    inParams.device = IN_DEV;
    inParams.channelCount = IN_CH;
    inParams.sampleFormat = paFloat32;
    inParams.suggestedLatency = Pa_GetDeviceInfo(inParams.device)->defaultHighInputLatency;

    err = Pa_OpenStream(&inStream,
                        &inParams,
                        nullptr,
                        SAMPLE_RATE,
                        FRAME,
                        paNoFlag,
                        nullptr, nullptr);
    if (err != paNoError)
        die_pa("Pa_OpenStream(in)", err);

    err = Pa_StartStream(inStream);
    if (err != paNoError)
        die_pa("Pa_StartStream(in)", err);
    std::cout << "\n\n";
    // ---------- Output stream ----------
    PaStream *outStream = nullptr;
    PaStreamParameters outParams{};
    outParams.device = OUT_DEV;
    outParams.channelCount = OUT_CH;
    outParams.sampleFormat = paFloat32;
    outParams.suggestedLatency = Pa_GetDeviceInfo(outParams.device)->defaultHighOutputLatency;

    err = Pa_OpenStream(&outStream,
                        nullptr,
                        &outParams,
                        SAMPLE_RATE,
                        FRAME,
                        paNoFlag,
                        nullptr, nullptr);
    if (err != paNoError)
        die_pa("Pa_OpenStream(out)", err);

    err = Pa_StartStream(outStream);
    if (err != paNoError)
        die_pa("Pa_StartStream(out)", err);

    // Print the actual sample rates PortAudio gave us
    const PaStreamInfo *inInfo = Pa_GetStreamInfo(inStream);
    const PaStreamInfo *outInfo = Pa_GetStreamInfo(outStream);
    // std::cout << "IN  opened SR = " << (inInfo ? inInfo->sampleRate : -1) << "\n";
    // std::cout << "OUT opened SR = " << (outInfo ? outInfo->sampleRate : -1) << "\n";
    std::cout << "Streaming... press Enter to stop.\n";

    std::vector<float> in((size_t)FRAME * (size_t)IN_CH);
    std::vector<float> play((size_t)FRAME * (size_t)OUT_CH);
    std::vector<float> play_next((size_t)FRAME);

    std::vector<spx_int16_t> rec16((size_t)FRAME);
    bool prev = false;
    std::vector<spx_int16_t> out16((size_t)FRAME);
    std::vector<spx_int16_t> ref16((size_t)FRAME, 0); // last frame actually played

    // echo and prepocess init
    const int TAIL = 4800;
    auto echo = speex_echo_state_init(FRAME, TAIL);
    int sr = SAMPLE_RATE;
    speex_echo_ctl(echo, SPEEX_ECHO_SET_SAMPLING_RATE, &sr);

    auto prep = speex_preprocess_state_init(FRAME, SAMPLE_RATE);
    speex_preprocess_ctl(prep, SPEEX_PREPROCESS_SET_ECHO_STATE, echo);

    int denoise = 1;
    speex_preprocess_ctl(prep, SPEEX_PREPROCESS_SET_DENOISE, &denoise);

    int echo_suppress = -30;
    speex_preprocess_ctl(prep, SPEEX_PREPROCESS_SET_ECHO_SUPPRESS, &echo_suppress);
    int echo_suppress_active = -15;
    speex_preprocess_ctl(prep, SPEEX_PREPROCESS_SET_ECHO_SUPPRESS_ACTIVE, &echo_suppress_active);

    int k = 0;
    while (run)
    {
        long rin = Pa_GetStreamReadAvailable(inStream);
        if (rin < (long)FRAME)
        {
            Pa_Sleep(1);
            continue;
        }
        long wout = Pa_GetStreamWriteAvailable(outStream);
        if (wout < (long)FRAME)
        {
            Pa_Sleep(1);
            continue;
        }

        PaError r = Pa_ReadStream(inStream, in.data(), FRAME);
        if (r != paNoError)
        {
            std::cerr << "Pa_ReadStream: " << Pa_GetErrorText(r) << " (" << r << ")\n";
            break;
        }

        // ch7 and ch8 rms
        if (++k % 200 == 0)
        {
            std::cout << "RMS ch7=" << rms_ch(6, in) << "  ch8=" << rms_ch(7, in) << "\n";
        }

        // Monitor mic channel 0 -> speaker mono
        for (int i = 0; i < FRAME; i++)
        {
            // mic to clean (ch0 here)
            float mic = in[(size_t)i * (size_t)IN_CH + 0];
            rec16[i] = f32_to_i16(mic);

            // far-end reference from loopback (ch7+ch8 -> mono)
            float ref = 0.5f * (in[(size_t)i * (size_t)IN_CH + 6] +
                                in[(size_t)i * (size_t)IN_CH + 7]);
            ref16[i] = f32_to_i16(ref);
        }

        speex_echo_playback(echo, ref16.data());
        // Cancel echo from mic capture
        speex_echo_capture(echo, rec16.data(), out16.data());
        speex_preprocess_run(prep, out16.data());

        // convert out16 to play, lower the gain
        for (int i = 0; i < FRAME; i++)
        {
            float y = i16_to_f32(out16[i]) * GAIN;

            if (y > 1.0f)
                y = 1.0f;
            if (y < -1.0f)
                y = -1.0f;

            play[i] = y;
        }

        PaError w = Pa_WriteStream(outStream, play.data(), FRAME);
        if (w != paNoError)
        {
            std::cerr << "Pa_WriteStream: " << Pa_GetErrorText(w) << " (" << w << ")\n";
            break;
        }
    }

    std::cout << "Stopping...\n";

    Pa_StopStream(outStream);
    Pa_CloseStream(outStream);
    Pa_StopStream(inStream);
    Pa_CloseStream(inStream);
    Pa_Terminate();

    if (stopper.joinable())
        stopper.join();
    return 0;
}
