// music_remove.cpp
// Input 8ch: mic1-6 + ref7-8
// Output 8ch .wav: clean mic1-6 + raw ref7 + raw ref8
//
// Build (example):
//  sudo apt install portaudio19-dev libwebrtc-audio-processing-dev
//  g++ -O3 -std=c++17 music_remove.cpp -o music_remove \
        $(pkg-config --cflags --libs webrtc-audio-processing-2 portaudio-2.0) \
        -lm -pthread
//
// Run:
//   PA_ALSA_PLUGHW=1 ./music_remove

#include <portaudio.h>
#include <sndfile.h>

#include <array>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include "api/audio/audio_processing.h"
#include "api/scoped_refptr.h"

constexpr int IN_DEV = 3;

constexpr int IN_CH = 8;
constexpr int MIC_CHS = 6; // chcd .1..ch6 -> indices 0..5
constexpr int REF_L = 6;   // ch7
constexpr int REF_R = 7;   // ch8
constexpr int OUT_CH = 8;  // clean6 + refL + refR

constexpr int SAMPLE_RATE = 48000;
constexpr int FRAME = 480; // 10 ms

static void die_pa(const char *where, PaError err)
{
    std::cerr << where << ": " << Pa_GetErrorText(err) << " (" << err << ")\n";
    std::exit(1);
}

static inline int16_t f32_to_i16(float x)
{
    if (x > 1.0f)
        x = 1.0f;
    if (x < -1.0f)
        x = -1.0f;
    return (int16_t)lrintf(x * 32767.0f);
}

static inline double rms_strided(const float *interleaved, int frames, int stride, int ch)
{
    double s = 0.0;
    for (int i = 0; i < frames; i++)
    {
        float v = interleaved[(size_t)i * (size_t)stride + (size_t)ch];
        s += (double)v * (double)v;
    }
    return std::sqrt(s / (double)frames);
}

static inline double rms_contig(const float *x, int n)
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

    // Input stream only
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

    // Output WAV: 8ch (clean6 + refL + refR)
    SF_INFO info{};
    info.samplerate = SAMPLE_RATE;
    info.channels = OUT_CH;
    info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SNDFILE *sf = sf_open("clean6_plus_ref78.wav", SFM_WRITE, &info);
    if (!sf)
    {
        std::cerr << "sf_open failed: " << sf_strerror(nullptr) << "\n";
        Pa_Terminate();
        return 1;
    }

    // WebRTC APM config
    webrtc::AudioProcessing::Config cfg;
    cfg.echo_canceller.enabled = true;
    cfg.echo_canceller.mobile_mode = false;
    cfg.high_pass_filter.enabled = true;

    cfg.noise_suppression.enabled = true;
    cfg.noise_suppression.level =
        webrtc::AudioProcessing::Config::NoiseSuppression::kHigh;

    cfg.gain_controller1.enabled = false;
    cfg.gain_controller2.enabled = false;

    webrtc::StreamConfig mono_cfg(SAMPLE_RATE, 1);
    webrtc::ProcessingConfig pc;
    pc.streams[webrtc::ProcessingConfig::kInputStream] = mono_cfg;
    pc.streams[webrtc::ProcessingConfig::kOutputStream] = mono_cfg;
    pc.streams[webrtc::ProcessingConfig::kReverseInputStream] = mono_cfg;
    pc.streams[webrtc::ProcessingConfig::kReverseOutputStream] = mono_cfg;

    // One APM per mic channel (simple and reliable)
    std::array<rtc::scoped_refptr<webrtc::AudioProcessing>, MIC_CHS> apm;
    for (int c = 0; c < MIC_CHS; c++)
    {
        apm[c] = webrtc::AudioProcessingBuilder().SetConfig(cfg).Create();
        apm[c]->Initialize(pc);
    }

    // Delay hint in ms.
    // If cancellation is weak, this being wrong is the #1 reason.
    // Start at 0; then replace with a calibrated value (see notes below).
    int delay_ms = 200;

    // Buffers
    std::vector<float> in((size_t)FRAME * (size_t)IN_CH);

    std::vector<float> ref_mono((size_t)FRAME);
    std::vector<float> ref_feed((size_t)FRAME); // boosted copy for AEC reverse

    std::vector<float> mic_f((size_t)FRAME);
    std::vector<float> out_f((size_t)FRAME);

    std::vector<float> clean6((size_t)FRAME * (size_t)MIC_CHS);
    std::vector<int16_t> out_pcm((size_t)FRAME * (size_t)OUT_CH);

    // Reference boost: compensates if ch7/8 is much quieter than the acoustic leak.
    float ref_gain = 1.0f;

    std::cout << "Recording clean6_plus_ref78.wav ... press Enter to stop.\n";

    int k = 0;
    while (run)
    {
        PaError r = Pa_ReadStream(inStream, in.data(), FRAME);
        if (r == paInputOverflowed)
            continue;
        if (r != paNoError)
        {
            std::cerr << "Pa_ReadStream: " << Pa_GetErrorText(r) << " (" << r << ")\n";
            break;
        }

        // Build mono reference from ch7+ch8
        for (int i = 0; i < FRAME; i++)
        {
            float L = in[(size_t)i * (size_t)IN_CH + (size_t)REF_L];
            float R = in[(size_t)i * (size_t)IN_CH + (size_t)REF_R];
            ref_mono[i] = 0.5f * (L + R);
        }

        // Update ref_gain slowly to roughly match mic-leak level (helps convergence)
        // Uses mic0 RMS as a proxy; clamps to avoid wild swings.
        double rms_ref = rms_contig(ref_mono.data(), FRAME);
        double rms_m0 = rms_strided(in.data(), FRAME, IN_CH, 0);
        float target = (rms_ref > 1e-9) ? (float)(rms_m0 / rms_ref) : 1.0f;
        target = std::clamp(target, 1.0f, 50.0f);
        ref_gain = 0.995f * ref_gain + 0.005f * target;

        // Process each mic channel 0..5
        for (int c = 0; c < MIC_CHS; c++)
        {
            // Reverse stream first (reference). Feed boosted reference.
            for (int i = 0; i < FRAME; i++)
                ref_feed[i] = ref_mono[i] * ref_gain;

            const float *rev_in[1] = {ref_feed.data()};
            float *rev_out[1] = {ref_feed.data()};
            apm[c]->ProcessReverseStream(rev_in, mono_cfg, mono_cfg, rev_out);

            // Forward stream (mic channel c)
            for (int i = 0; i < FRAME; i++)
                mic_f[i] = in[(size_t)i * (size_t)IN_CH + (size_t)c];

            apm[c]->set_stream_delay_ms(delay_ms);

            const float *cap_in[1] = {mic_f.data()};
            float *cap_out[1] = {out_f.data()};
            int pr = apm[c]->ProcessStream(cap_in, mono_cfg, mono_cfg, cap_out);

            if (pr != webrtc::AudioProcessing::kNoError)
            {
                for (int i = 0; i < FRAME; i++)
                    out_f[i] = mic_f[i];
            }

            for (int i = 0; i < FRAME; i++)
                clean6[(size_t)i * (size_t)MIC_CHS + (size_t)c] = out_f[i];
        }

        // Interleave output: [clean1..clean6, raw ch7, raw ch8]
        for (int i = 0; i < FRAME; i++)
        {
            for (int c = 0; c < MIC_CHS; c++)
            {
                float x = clean6[(size_t)i * (size_t)MIC_CHS + (size_t)c];
                out_pcm[(size_t)i * (size_t)OUT_CH + (size_t)c] = f32_to_i16(x);
            }

            float raw7 = in[(size_t)i * (size_t)IN_CH + (size_t)REF_L];
            float raw8 = in[(size_t)i * (size_t)IN_CH + (size_t)REF_R];
            out_pcm[(size_t)i * (size_t)OUT_CH + 6] = f32_to_i16(raw7);
            out_pcm[(size_t)i * (size_t)OUT_CH + 7] = f32_to_i16(raw8);
        }

        sf_writef_short(sf, out_pcm.data(), FRAME);

        if (++k % 200 == 0)
        {
            double r_ref = rms_contig(ref_mono.data(), FRAME);
            double r_c1 = rms_contig(&clean6[0], FRAME); // clean mic1 contiguous
            std::cout << "rms_ref=" << r_ref
                      << " rms_clean_ch1=" << r_c1
                      << " ref_gain=" << ref_gain
                      << " delay_ms=" << delay_ms
                      << "\n";
        }
    }

    sf_close(sf);

    Pa_StopStream(inStream);
    Pa_CloseStream(inStream);
    Pa_Terminate();

    if (stopper.joinable())
        stopper.join();
    return 0;
}
