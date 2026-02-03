// realtime_tdoa.cpp

// Build (Linux example):
// g++ -O3 -std=c++17 realtime_tdoa.cpp -o realtime_tdoa \
    $(pkg-config --cflags --libs portaudio-2.0 rnnoise fftw3 webrtc-audio-processing-2) \
    -lm -pthread
//
// Run:
// PA_ALSA_PLUGHW=1 ./realtime_tdoa
//
// What it does (minimal + practical):
// - Captures 8ch @48kHz
// - (if enabled) RNNoise denoise on ch0..ch5 -> ring buffer -> GCC-PHAT DOA (same as your TDOA code)
// - (if enabled) WebRTC AEC3 on mono monitor path (mic ch0) + optional RNNoise on that monitor output
// - Plays mono output to speaker

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include <portaudio.h>
#include <fftw3.h>
#include "rnnoise.h"

#include "api/audio/audio_processing.h"
// #include "api/audio/audio_processing_builder.h"
#include "api/scoped_refptr.h"

static constexpr double PI = 3.14159265358979323846;

// -------------------- PortAudio constants --------------------
constexpr int IN_DEV = 3;  // input device index
constexpr int OUT_DEV = 4; // output device index: 2 = headphone, 4 = speaker

constexpr int IN_CH = 8;  // number of input channels
constexpr int OUT_CH = 1; // number of output channels

constexpr int SAMPLE_RATE = 48000; // input sample rate
constexpr int FRAME = 480;         // number of frames in the buffer

constexpr float GAIN = 0.2f;       // mic gain
constexpr float LIM_THRESH = 0.1f; // limiter threshold

constexpr bool USE_AEC = true; // AEC toggle
constexpr bool USE_NS = true;  // Noise suppression toggle

// -------------------- DOA constants -----------------
constexpr int M = 6;                 // number of input channels using
constexpr int N = 512;               // frameLen for GCC
constexpr double C = 343.0;          // speed of sound
constexpr double OFFSET = 150.0;     // degree offset (for aligning middle of mic 1 & 6 as 0 deg)
constexpr double STEP_DEG = 0.5;     // angle checking step size
constexpr int N_THETA = 360.0 / 0.5; // number of angles
constexpr double OK_FACTOR = 1.20;   // range of acceptable angles

constexpr int RING = SAMPLE_RATE; // ring buffer sample rate
constexpr int SEARCH = 4800;      // required samples for TDOA
constexpr float PEAK_THRESH = 0.02f;     // TDOA peak theshold

constexpr uint64_t UPDATE_EVERY = 30; // frames

// --------------- Double talk constants --------------
constexpr float RENDER_TH = 0.0001f;  // speaker active if rms(render) > this
constexpr float MIC_TH = 0.005f;      // ignore quiet mic
constexpr float NEAR_RATIO = 1000.0f; // mic must dominate render by this factor
constexpr float ECHO_CORR_TH = 0.4f;  // echo-likeness threshold (normalized corr peak)

constexpr int RENDER_RING = SAMPLE_RATE; // 1s render history

static void err_pa(const char *where, PaError err)
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

    const float attack = 0.2f;
    const float release = 0.98f;

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

static inline double mod360(double deg)
{
    double y = std::fmod(deg, 360.0);
    if (y < 0)
        y += 360.0;
    return y;
}

static inline float echo_corr_peak(const float *mic, int n,
                                   const std::vector<float> &rr, int ringN, int wpos,
                                   int centerDelaySamp, int slackSamp, int stepSamp)
{
    float best = 0.0f;

    int d0 = std::max(0, centerDelaySamp - slackSamp);
    int d1 = std::min(ringN - n - 1, centerDelaySamp + slackSamp);

    for (int d = d0; d <= d1; d += stepSamp)
    {
        double dot = 0.0, em = 0.0, er = 0.0;

        // compare current mic frame with render frame that occurred d samples earlier
        int base = wpos - d - n;
        for (int i = 0; i < n; i++)
        {
            int idx = base + i;
            idx %= ringN;
            if (idx < 0)
                idx += ringN;

            float r = rr[(size_t)idx];
            float m = mic[i];

            dot += (double)m * (double)r;
            em += (double)m * (double)m;
            er += (double)r * (double)r;
        }

        double den = std::sqrt(em * er) + 1e-12;
        float c = (float)std::fabs(dot / den);
        if (c > best)
            best = c;
    }

    return best;
}

static inline float render_rms_at_delay(const std::vector<float> &rr, int ringN, int wpos,
                                        int delaySamp, int n)
{
    double s = 0.0;
    int base = wpos - delaySamp - n; // same alignment as echo_corr_peak uses for d=delaySamp
    for (int i = 0; i < n; i++)
    {
        int idx = base + i;
        idx %= ringN;
        if (idx < 0)
            idx += ringN;
        float r = rr[(size_t)idx];
        s += (double)r * (double)r;
    }
    return (float)std::sqrt(s / (double)n);
}

struct PairMeas
{
    int i, j;
    double tau;
    double w;
};

class GCCPhat
{
public:
    GCCPhat(int N, double fs) : N_(N), fs_(fs)
    {
        nfft_ = 1;
        while (nfft_ < 2 * N_ - 1)
            nfft_ <<= 1;
        in1_ = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * nfft_);
        in2_ = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * nfft_);
        out1_ = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * nfft_);
        out2_ = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * nfft_);
        X_ = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * nfft_);
        x_ = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * nfft_);
        if (!in1_ || !in2_ || !out1_ || !out2_ || !X_ || !x_)
            throw std::runtime_error("fftw_malloc failed");
        p1_ = fftw_plan_dft_1d(nfft_, in1_, out1_, FFTW_FORWARD, FFTW_MEASURE);
        p2_ = fftw_plan_dft_1d(nfft_, in2_, out2_, FFTW_FORWARD, FFTW_MEASURE);
        pi_ = fftw_plan_dft_1d(nfft_, X_, x_, FFTW_BACKWARD, FFTW_MEASURE);
    }

    ~GCCPhat()
    {
        if (p1_)
            fftw_destroy_plan(p1_);
        if (p2_)
            fftw_destroy_plan(p2_);
        if (pi_)
            fftw_destroy_plan(pi_);
        if (in1_)
            fftw_free(in1_);
        if (in2_)
            fftw_free(in2_);
        if (out1_)
            fftw_free(out1_);
        if (out2_)
            fftw_free(out2_);
        if (X_)
            fftw_free(X_);
        if (x_)
            fftw_free(x_);
    }

    // GCC-PHAT process
    PairMeas process(const std::vector<float> &a, const std::vector<float> &b, int i, int j, int maxLag)
    {
        if ((int)a.size() != N_ || (int)b.size() != N_)
            throw std::runtime_error("GCC input size != N");

        // FFT on both signals
        for (int k = 0; k < nfft_; k++)
        {
            in1_[k][0] = (k < N_) ? a[k] : 0.0;
            in1_[k][1] = 0.0;
            in2_[k][0] = (k < N_) ? b[k] : 0.0;
            in2_[k][1] = 0.0;
        }
        fftw_execute(p1_);
        fftw_execute(p2_);

        // cross-power spectrum + normalize
        const double eps = 1e-12;
        for (int k = 0; k < nfft_; k++)
        {
            double ar = out1_[k][0], ai = out1_[k][1];
            double br = out2_[k][0], bi = out2_[k][1];
            double re = ar * br + ai * bi;
            double im = ai * br - ar * bi;
            double mag = std::sqrt(re * re + im * im) + eps;
            X_[k][0] = re / mag;
            X_[k][1] = im / mag;
        }

        // inverse FFT
        fftw_execute(pi_);

        maxLag = std::min(maxLag, N_ - 1);
        double pk = -1.0, sumA = 0.0;
        int cnt = 0, bestLag = 0;
        // retrieve time delay
        for (int lag = -maxLag; lag <= maxLag; lag++)
        {
            int idx = (lag < 0) ? (nfft_ + lag) : lag;
            double r = x_[idx][0] / nfft_;
            double A = std::abs(r);
            sumA += A;
            cnt++;
            if (A > pk)
            {
                pk = A;
                bestLag = lag;
            }
        }

        double meanA = (cnt > 0) ? (sumA / cnt) : 1e-12;
        double peakRatio = pk / (meanA + 1e-12);
        return {i, j, (double)bestLag / fs_, peakRatio};
    }

private:
    int N_ = 0, nfft_ = 0;
    double fs_ = 0;
    fftw_complex *in1_ = nullptr, *in2_ = nullptr, *out1_ = nullptr, *out2_ = nullptr, *X_ = nullptr, *x_ = nullptr;
    fftw_plan p1_ = nullptr, p2_ = nullptr, pi_ = nullptr;
};

struct PairInfo
{
    int i, j;
    int maxLag;
    double dpx, dpy; // (pi - pj)
};

class DoaSolver
{
public:
    DoaSolver() : gcc_(N, (double)SAMPLE_RATE)
    {
        hann_.resize(N);
        for (int n = 0; n < N; n++)
            hann_[n] = 0.5 * (1.0 - std::cos(2.0 * PI * n / (N - 1)));

        // hard-coded mic pairing distances
        double d[M][M] = {};
        auto setD = [&](int a, int b, double meters)
        { d[a][b] = d[b][a] = meters; };
        setD(0, 1, 35e-3);
        setD(0, 2, 60e-3);
        setD(0, 3, 69.462e-3);
        setD(0, 4, 60e-3);
        setD(0, 5, 35e-3);
        setD(1, 2, 35e-3);
        setD(1, 3, 60e-3);
        setD(1, 4, 69.462e-3);
        setD(1, 5, 60e-3);
        setD(2, 3, 35e-3);
        setD(2, 4, 60e-3);
        setD(2, 5, 69.462e-3);
        setD(3, 4, 35e-3);
        setD(3, 5, 60e-3);
        setD(4, 5, 35e-3);

        double dia = std::sqrt(60e-3 * 60e-3 + 35e-3 * 35e-3);
        double R = dia / 2.0;
        double angDeg[M] = {0, 60, 120, 180, 240, 300};
        for (int i = 0; i < M; i++)
        {
            double a = angDeg[i] * PI / 180.0;
            px_[i] = R * std::cos(a);
            py_[i] = R * std::sin(a);
        }

        int k = 0;
        for (int i = 0; i < M; i++)
        {
            for (int j = i + 1; j < M; j++)
            {
                pairs_[k].i = i;
                pairs_[k].j = j;
                pairs_[k].maxLag = (int)std::ceil((d[i][j] / C) * SAMPLE_RATE);
                pairs_[k].dpx = px_[i] - px_[j];
                pairs_[k].dpy = py_[i] - py_[j];
                k++;
            }
        }

        for (int ch = 0; ch < M; ch++)
        {
            xSearch_[ch].assign(SEARCH, 0.0f);
            xWin_[ch].assign(N, 0.0f);
        }
        ok_.assign(N_THETA, 0);
    }

    bool compute(const std::vector<float> &ring, int writePos, long long totalWritten, double &thetaBest, int &Lidx, int &Ridx)
    {
        if (totalWritten < SEARCH)
            return false;

        // copy signal to xSearch
        for (int n = 0; n < SEARCH; n++)
        {
            int pos = (writePos - SEARCH + n) % RING;
            if (pos < 0)
                pos += RING;
            for (int ch = 0; ch < M; ch++)
                xSearch_[ch][n] = ring[(size_t)ch * (size_t)RING + (size_t)pos];
        }

        // find peak in sample
        int kPeak = 0;
        double best = -1.0;
        for (int n = 0; n < SEARCH; n++)
        {
            double env = 0.0;
            for (int ch = 0; ch < M; ch++)
                env = std::max(env, (double)std::abs(xSearch_[ch][n]));
            if (env > best)
            {
                best = env;
                kPeak = n;
            }
        }
        if ((float)best < PEAK_THRESH)
            return false;

        // retrieve window around peak
        int startK = kPeak - N / 2;
        startK = std::max(0, std::min(startK, SEARCH - N));
        for (int ch = 0; ch < M; ch++)
            std::copy_n(xSearch_[ch].begin() + startK, N, xWin_[ch].begin());

        // DC removal + Hann window (whatever that means)
        for (int ch = 0; ch < M; ch++)
        {
            double mean = 0.0;
            for (float v : xWin_[ch])
                mean += v;
            mean /= N;
            for (int n = 0; n < N; n++)
                xWin_[ch][n] = (float)((xWin_[ch][n] - mean) * hann_[n]);
        }

        // Computer GCC-PHAT to get delay for each pair
        double wMax = 1e-12;
        for (size_t k = 0; k < pairs_.size(); k++)
        {
            auto &p = pairs_[k];
            auto m = gcc_.process(xWin_[p.i], xWin_[p.j], p.i, p.j, p.maxLag);
            tau_[k] = m.tau;
            w_[k] = m.w;
            wMax = std::max(wMax, m.w);
        }
        for (double &w : w_)
            w /= wMax;

        // find angle best matches the delays of each pair
        auto sseForTheta = [&](int t)
        {
            double th = (t * STEP_DEG) * PI / 180.0;
            double ux = std::cos(th), uy = std::sin(th);
            double sse = 0.0;
            for (size_t k = 0; k < pairs_.size(); k++)
            {
                double tauPred = (pairs_[k].dpx * ux + pairs_[k].dpy * uy) / C;
                double e = tau_[k] - tauPred;
                double we = w_[k] * e;
                sse += we * we;
            }
            return sse;
        };

        // find minimum sse
        int bestIdx = 0;
        double errMin = 1e300;
        for (int t = 0; t < N_THETA; t++)
        {
            double sse = sseForTheta(t);
            if (sse < errMin)
            {
                errMin = sse;
                bestIdx = t;
            }
        }

        // mark range of angle
        double thresh = errMin * OK_FACTOR;
        for (int t = 0; t < N_THETA; t++)
            ok_[t] = (sseForTheta(t) <= thresh);

        // return Lidx and Ridx around bestIdx
        Lidx = Ridx = bestIdx;
        if (!ok_[bestIdx])
        {
            Lidx = Ridx = -1;
        }
        else
        {
            while (true)
            {
                int nl = (Lidx - 1 + N_THETA) % N_THETA;
                if (nl == bestIdx || !ok_[nl])
                    break;
                Lidx = nl;
            }
            while (true)
            {
                int nr = (Ridx + 1) % N_THETA;
                if (nr == bestIdx || !ok_[nr])
                    break;
                Ridx = nr;
            }
        }

        // convert bestIdx to deg angle
        thetaBest = mod360(bestIdx * STEP_DEG - OFFSET);
        return true;
    }

private:
    GCCPhat gcc_;

    std::vector<double> hann_;
    std::array<double, M> px_{};
    std::array<double, M> py_{};

    std::array<PairInfo, 15> pairs_{};
    std::array<double, 15> tau_{};
    std::array<double, 15> w_{};

    std::array<std::vector<float>, M> xSearch_;
    std::array<std::vector<float>, M> xWin_;
    std::vector<uint8_t> ok_;
};

int main()
{
    PaError err = Pa_Initialize();
    if (err != paNoError)
        err_pa("Pa_Initialize", err);

    std::cout << "\n\n"; // separate stderr

    std::atomic<bool> run{true};
    std::thread stopper([&]
                        {
                            std::string line;
                            if (std::getline(std::cin, line))
                                run = false; });

    // ---------- Input stream ----------
    PaStream *inStream = nullptr;
    PaStreamParameters inParams{};
    inParams.device = IN_DEV;
    inParams.channelCount = IN_CH;
    inParams.sampleFormat = paFloat32;
    inParams.suggestedLatency = Pa_GetDeviceInfo(inParams.device)->defaultHighInputLatency;
    inParams.hostApiSpecificStreamInfo = nullptr;

    err = Pa_OpenStream(&inStream,
                        &inParams,
                        nullptr,
                        SAMPLE_RATE,
                        FRAME,
                        paNoFlag,
                        nullptr, nullptr);
    if (err != paNoError)
        err_pa("Pa_OpenStream(in)", err);

    err = Pa_StartStream(inStream);
    if (err != paNoError)
        err_pa("Pa_StartStream(in)", err);

    // ---------- Output stream ----------
    PaStream *outStream = nullptr;
    PaStreamParameters outParams{};
    outParams.device = OUT_DEV;
    outParams.channelCount = OUT_CH;
    outParams.sampleFormat = paFloat32;

    auto outDevInfo = Pa_GetDeviceInfo(outParams.device);
    double want = 0.20;
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
        err_pa("Pa_OpenStream(out)", err);

    err = Pa_StartStream(outStream);
    if (err != paNoError)
        err_pa("Pa_StartStream(out)", err);

    const PaStreamInfo *inInfo = Pa_GetStreamInfo(inStream);
    const PaStreamInfo *outInfo = Pa_GetStreamInfo(outStream);

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
    webrtc::AudioProcessing::Config cfg;
    cfg.echo_canceller.enabled = true;
    cfg.echo_canceller.mobile_mode = false;
    cfg.high_pass_filter.enabled = true;

    cfg.noise_suppression.enabled = false;

    cfg.gain_controller1.enabled = false;
    cfg.gain_controller2.enabled = false;

    rtc::scoped_refptr<webrtc::AudioProcessing> apm =
        webrtc::AudioProcessingBuilder().SetConfig(cfg).Create();

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

    std::vector<float> render_prev((size_t)FRAME, 0.0f);
    std::vector<float> render_ring((size_t)RENDER_RING, 0.0f);
    int render_wpos = 0;
    long long render_total = 0;

    float lim_gain = 1.0f;

    // ---------------- TDOA buffers ----------------
    std::vector<float> ring((size_t)M * (size_t)RING, 0.0f);
    int writePos = 0;
    long long totalWritten = 0;

    std::vector<float> den((size_t)M * (size_t)FRAME);

    std::vector<float> tmp(FRAME);
    std::vector<float> tmp16(FRAME);
    std::vector<float> out16(FRAME);

    DoaSolver solver;

    // ---------------- RNNoise states ----------------
    std::array<DenoiseState *, M> st{};
    DenoiseState *st_out = nullptr;

    if (USE_NS)
    {
        for (int ch = 0; ch < M; ch++)
        {
            st[ch] = rnnoise_create(nullptr);
            if (!st[ch])
                return 1;
        }
        st_out = rnnoise_create(nullptr);
        if (!st_out)
            return 1;
    }

    std::cout << "AEC=" << (USE_AEC ? "ON" : "OFF")
              << " NS=" << (USE_NS ? "ON" : "OFF")
              << " DOA=ON (GCC-PHAT) Press Enter to stop.\n";

    uint64_t k = 0;
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

        // RNNoise on ch0..5 (TDOA path) or else normal
        if (USE_NS)
        {
            for (int ch = 0; ch < M; ch++)
            {
                for (int i = 0; i < FRAME; i++)
                    tmp[i] = in[(size_t)i * (size_t)IN_CH + (size_t)ch];
                for (int i = 0; i < FRAME; i++)
                    tmp16[i] = tmp[i] * 32768.0f;

                rnnoise_process_frame(st[ch], out16.data(), tmp16.data());

                for (int i = 0; i < FRAME; i++)
                    den[(size_t)ch * (size_t)FRAME + (size_t)i] = out16[i] / 32768.0f;
            }
        }
        else
        {
            for (int ch = 0; ch < M; ch++)
                for (int i = 0; i < FRAME; i++)
                    den[(size_t)ch * (size_t)FRAME + (size_t)i] = in[(size_t)i * (size_t)IN_CH + (size_t)ch];
        }

        // write denoised into ring (time-aligned)
        for (int i = 0; i < FRAME; i++)
        {
            int pos = (writePos + i) % RING;
            for (int ch = 0; ch < M; ch++)
                ring[(size_t)ch * (size_t)RING + (size_t)pos] = den[(size_t)ch * (size_t)FRAME + (size_t)i];
        }
        writePos = (writePos + FRAME) % RING;
        totalWritten += FRAME;

        // Feed the reverse/render stream (AEC reference)
        if (USE_AEC)
        {
            const float *rev_in[1] = {render_prev.data()};
            float *rev_out[1] = {render_prev.data()};
            int er = apm->ProcessReverseStream(rev_in, mono_cfg, mono_cfg, rev_out);
            if (er != webrtc::AudioProcessing::kNoError)
            {
                if (k % 200 == 0)
                    std::cerr << "ProcessReverseStream err=" << er << "\n";
            }
        }

        // Extract mic channel 0 into mono buffer (monitor path)
        for (int i = 0; i < FRAME; i++)
            mic_f[i] = in[(size_t)i * (size_t)IN_CH + 0];

        // Process capture stream (AEC)
        const float *cap_in[1] = {mic_f.data()};
        float *cap_out[1] = {out_f.data()};

        if (USE_AEC)
            apm->set_stream_delay_ms(delay_ms);

        int pr = apm->ProcessStream(cap_in, mono_cfg, mono_cfg, cap_out);
        if (pr != webrtc::AudioProcessing::kNoError)
        {
            for (int i = 0; i < FRAME; i++)
                out_f[i] = mic_f[i];
            if (k % 200 == 0)
                std::cerr << "ProcessStream err=" << pr << "\n";
        }

        // Optional: RNNoise on the monitor output (after AEC)
        if (USE_NS)
        {
            for (int i = 0; i < FRAME; i++)
                tmp16[i] = out_f[i] * 32768.0f;
            rnnoise_process_frame(st_out, out16.data(), tmp16.data());
            for (int i = 0; i < FRAME; i++)
                out_f[i] = out16[i] / 32768.0f;
        }

        // Build speaker buffer (apply gain -> apply limiter -> prevent clip)
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

        for (int i = 0; i < FRAME; i++)
            render_ring[(size_t)((render_wpos + i) % RENDER_RING)] = play[i];
        render_wpos = (render_wpos + FRAME) % RENDER_RING;
        render_total += FRAME;

        // Save actual output as next frame's AEC reference
        for (int i = 0; i < FRAME; i++)
            render_prev[i] = play[i];

        // if (++k % 100 == 0)
        // {
        //     std::cout << "RMS play=" << rms_f32_frame(play.data(), FRAME) << "\n";
        // }

        PaError w = Pa_WriteStream(outStream, play.data(), FRAME);
        if (w == paOutputUnderflowed)
            continue;
        if (w != paNoError)
        {
            std::cerr << "Pa_WriteStream: " << Pa_GetErrorText(w) << " (" << w << ")\n";
            break;
        }

        // DOA update every UPDATE_EVERY
        if (++k % UPDATE_EVERY)
            continue;

        int center = (int)lrint((delay_ms / 1000.0) * SAMPLE_RATE);
        int slack = 4 * FRAME; // +/- 40ms search window
        int step = 16;         // CPU reduction

        // render_rms must refer to the SAME delayed render segment used by correlation
        float render_rms = 0.0f;
        if (render_total > (FRAME + center))
            render_rms = render_rms_at_delay(render_ring, RENDER_RING, render_wpos, center, FRAME);

        float mic_rms = (float)rms_f32_frame(mic_f.data(), FRAME);

        bool speaker_active = (render_rms > RENDER_TH);
        bool near_dominant = (mic_rms > MIC_TH) && (mic_rms > NEAR_RATIO * (render_rms + 1e-6f));

        float corrpk = 0.0f;
        bool echo_like = false;

        if (speaker_active && render_total > RENDER_RING / 2)
        {
            corrpk = echo_corr_peak(mic_f.data(), FRAME,
                                    render_ring, RENDER_RING, render_wpos,
                                    center, slack, step);
            echo_like = (corrpk > ECHO_CORR_TH);
        }

        // static uint64_t dbg = 0;
        // if ((dbg++ % 1) == 0) // change %1 to %5 to print less often
        // {
        //     std::cout << "=============================\n";
        //     std::cout
        //         << "[DBG] mic_rms=" << mic_rms
        //         << " render_rms=" << render_rms
        //         << " ratio=" << (mic_rms / (render_rms + 1e-6f))
        //         << " corrpk=" << corrpk
        //         << " speaker_active=" << speaker_active
        //         << " near_dominant=" << near_dominant
        //         << " echo_like=" << echo_like
        //         << "\n";
        //     std::cout << "=============================\n";
        // }

        // If speaker is active and mic looks like echo, block DOA unless near-end clearly dominates
        if (speaker_active && echo_like && !near_dominant)
            continue;

        double thetaBest = 0.0;
        int Lidx = -1, Ridx = -1;
        if (!solver.compute(ring, writePos, totalWritten, thetaBest, Lidx, Ridx))
            continue;

        std::cout << "DOA = " << thetaBest << " deg";

        double Lout = mod360(Lidx * STEP_DEG - OFFSET);
        double Rout = mod360(Ridx * STEP_DEG - OFFSET);
        std::cout << "  Range = " << Lout << " .. " << Rout << " deg";

        std::cout << "\n";
    }

    std::cout << "Stopping...\n";

    if (USE_NS)
    {
        for (auto *p : st)
            rnnoise_destroy(p);
        rnnoise_destroy(st_out);
    }

    Pa_StopStream(outStream);
    Pa_CloseStream(outStream);
    Pa_StopStream(inStream);
    Pa_CloseStream(inStream);
    Pa_Terminate();

    if (stopper.joinable())
        stopper.join();

    return 0;
}
