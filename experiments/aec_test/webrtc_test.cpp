// webrtc_test.cpp
//
// Install/Build:
//  sudo apt install portaudio19-dev libwebrtc-audio-processing-dev
//  g++ -O3 -std=c++17 webrtc_test.cpp -o webrtc_test \
        $(pkg-config --cflags --libs webrtc-audio-processing-2 portaudio-2.0) \
        -lm -pthread
// Run:   PA_ALSA_PLUGHW=1 ./webrtc_test
//
// Notes:
// - WebRTC AEC3 expects 10 ms frames. FRAME=480 @ 48 kHz is correct.
// - AEC is designed to remove *rendered* audio leaking into the mic. If your
//   speaker is literally playing your mic (talkback), AEC can still reduce
//   delayed “echo/loop” paths, but it can also start cancelling your own voice
//   if the system behaves like a closed loop. Keep GAIN low + keep your limiter.

#include <portaudio.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "api/audio/audio_processing.h"
// #include "api/audio/audio_processing_builder.h"
#include "api/scoped_refptr.h" // for rtc::scoped_refptr

constexpr int IN_DEV = 3;
constexpr int OUT_DEV = 4;

constexpr int IN_CH = 8;
constexpr int OUT_CH = 1;

constexpr int SAMPLE_RATE = 48000;
constexpr int FRAME = 480;

constexpr float GAIN = 0.1f;
constexpr float LIM_THRESH = 0.05f; // peak limiter threshold (amplitude, not RMS)

constexpr bool USE_AEC = true;

static void die_pa(const char *where, PaError err)
{
    std::cerr << where << ": " << Pa_GetErrorText(err) << " (" << err << ")\n";
    std::exit(1);
}

static inline void apply_limiter(float *y, int n,
                                 float threshold,
                                 float &gain_state)
{
    float peak = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float a = std::fabs(y[i]);
        if (a > peak)
            peak = a;
    }

    float target = 1.0f;
    if (peak > threshold && peak > 1e-12f)
        target = threshold / peak;

    const float attack = 0.2f;   // smaller = faster clamp
    const float release = 0.98f; // closer to 1 = slower recovery

    if (target < gain_state)
        gain_state = attack * gain_state + (1.0f - attack) * target;
    else
        gain_state = release * gain_state + (1.0f - release) * target;

    for (int i = 0; i < n; i++)
        y[i] *= gain_state;
}

static inline double rms_f32_frame(const float *x, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += (double)x[i] * (double)x[i];
    return std::sqrt(s / (double)n);
}

int main()
{
    PaError err = Pa_Initialize();
    if (err != paNoError)
        die_pa("Pa_Initialize", err);

    std::cout << "\n\n";
    
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

    // ---------- Output stream ----------
    PaStream *outStream = nullptr;
    PaStreamParameters outParams{};
    outParams.device = OUT_DEV;
    outParams.channelCount = OUT_CH;
    outParams.sampleFormat = paFloat32;

    auto outDevInfo = Pa_GetDeviceInfo(outParams.device);
    double want = 0.20; // 200 ms
    outParams.suggestedLatency = (outDevInfo->defaultHighOutputLatency > want)
                                     ? outDevInfo->defaultHighOutputLatency
                                     : want;

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

    const PaStreamInfo *inInfo = Pa_GetStreamInfo(inStream);
    const PaStreamInfo *outInfo = Pa_GetStreamInfo(outStream);

    // Approx delay (ms) between render (reverse) and capture (near-end).
    // (This is only an estimate; DelayAgnostic helps when enabled.)
    int delay_ms = 0;
    if (inInfo && outInfo)
    {
        double d = (inInfo->inputLatency + outInfo->outputLatency) * 1000.0;
        if (d < 0)
            d = 0;
        delay_ms = (int)lrint(d);
    }

    std::cout << "Streaming... press Enter to stop. delay_ms~" << delay_ms << "\n";

    // ---------------- WebRTC AudioProcessing (AEC3) ----------------
    // Enable AEC3 + delay-agnostic mode via ExtraOptions.
    webrtc::AudioProcessing::Config cfg;
    cfg.echo_canceller.enabled = true;
    cfg.echo_canceller.mobile_mode = false; // non-mobile path
    cfg.high_pass_filter.enabled = true;

    cfg.noise_suppression.enabled = true;
    cfg.noise_suppression.level =
        webrtc::AudioProcessing::Config::NoiseSuppression::kHigh;

    // If you don’t want AGC, keep these off:
    cfg.gain_controller1.enabled = false;
    cfg.gain_controller2.enabled = false;

    // APM is ref-counted (not std::unique_ptr)
    rtc::scoped_refptr<webrtc::AudioProcessing> apm =
        webrtc::AudioProcessingBuilder().SetConfig(cfg).Create();

    // Float interface configs.
    webrtc::StreamConfig mono_cfg(SAMPLE_RATE, 1);

    webrtc::ProcessingConfig pc;
    pc.streams[webrtc::ProcessingConfig::kInputStream] = mono_cfg;
    pc.streams[webrtc::ProcessingConfig::kOutputStream] = mono_cfg;
    pc.streams[webrtc::ProcessingConfig::kReverseInputStream] = mono_cfg;
    pc.streams[webrtc::ProcessingConfig::kReverseOutputStream] = mono_cfg;
    apm->Initialize(pc);

    // ---------------- Buffers ----------------
    std::vector<float> in((size_t)FRAME * (size_t)IN_CH);
    std::vector<float> play((size_t)FRAME * (size_t)OUT_CH);

    std::vector<float> mic_f((size_t)FRAME);
    std::vector<float> out_f((size_t)FRAME);

    // render_prev = the *actual* samples we sent to the speaker last frame.
    // We feed this to ProcessReverseStream as the AEC reference.
    std::vector<float> render_prev((size_t)FRAME, 0.0f);

    float lim_gain = 1.0f;
    int k = 0;

    while (run)
    {
        PaError r = Pa_ReadStream(inStream, in.data(), FRAME);
        if (r == paInputOverflowed)
        {
            std::cout << "overflow\n";
            continue;
        }
        if (r != paNoError)
        {
            std::cerr << "Pa_ReadStream: " << Pa_GetErrorText(r) << " (" << r << ")\n";
            break;
        }

        // 1) Feed the reverse/render stream (AEC reference).
        //    Using last frame's actual output keeps timing consistent.
        if (USE_AEC)
        {
            const float *rev_in[1] = {render_prev.data()};
            float *rev_out[1] = {render_prev.data()}; // in-place OK
            int er = apm->ProcessReverseStream(rev_in, mono_cfg, mono_cfg, rev_out);
            if (er != webrtc::AudioProcessing::kNoError)
            {
                // Non-fatal; keep going.
                if (k % 200 == 0)
                    std::cerr << "ProcessReverseStream err=" << er << "\n";
            }
        }

        // 2) Extract mic channel 0 into mono buffer.
        for (int i = 0; i < FRAME; i++)
            mic_f[i] = in[(size_t)i * (size_t)IN_CH + 0];

        // 3) Process capture stream.
        const float *cap_in[1] = {mic_f.data()};
        float *cap_out[1] = {out_f.data()};

        if (USE_AEC)
            apm->set_stream_delay_ms(delay_ms); // must be set when echo processing is enabled

        int pr = apm->ProcessStream(cap_in, mono_cfg, mono_cfg, cap_out);
        if (pr != webrtc::AudioProcessing::kNoError)
        {
            // If APM errors, fall back to raw mic for that frame.
            for (int i = 0; i < FRAME; i++)
                out_f[i] = mic_f[i];

            if (k % 200 == 0)
                std::cerr << "ProcessStream err=" << pr << "\n";
        }

        // 4) Build speaker buffer (gain -> limiter -> clip).
        for (int i = 0; i < FRAME; i++)
            play[i] = out_f[i] * GAIN;

        apply_limiter(play.data(), FRAME, LIM_THRESH, lim_gain);

        for (int i = 0; i < FRAME; i++)
        {
            float y = play[i];
            if (y > 1.0f)
                y = 1.0f;
            if (y < -1.0f)
                y = -1.0f;
            play[i] = y;
        }

        // 5) Save the *actual output* as next frame's AEC reference.
        for (int i = 0; i < FRAME; i++)
            render_prev[i] = play[i];

        if (++k % 100 == 0)
            std::cout << "RMS play=" << rms_f32_frame(play.data(), FRAME) << "\n";

        PaError w = Pa_WriteStream(outStream, play.data(), FRAME);
        if (w == paOutputUnderflowed)
        {
            std::cout << "underflow\n";
            continue;
        }
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
