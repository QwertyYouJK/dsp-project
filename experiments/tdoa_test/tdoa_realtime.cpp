// tdoa_realtime.cpp
// Build (Linux example):
// g++ -O3 -std=c++17 tdoa_realtime.cpp -o tdoa_realtime \
    $(pkg-config --cflags --libs portaudio-2.0 fftw3) \
    -lm -pthread
// Run:
// PA_ALSA_PLUGHW=1 ./tdoa_realtime

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include <portaudio.h>
#include <fftw3.h>
#include "rnnoise.h"

static constexpr double PI = 3.14159265358979323846;

// -------------------- Audio / stream constants --------------------
constexpr int SAMPLE_RATE = 48000;
constexpr int IN_CH = 8;
// constexpr int OUT_CH = 1;
constexpr int FRAME = 480;
constexpr int DEVICE_INDEX = 3;

// -------------------- DOA constants (same values) -----------------
constexpr int M = 6;        // use ch0..ch5
constexpr int N = 512;      // frameLen for GCC
constexpr double C = 343.0; // speed of sound
constexpr double OFFSET = 150.0;
constexpr double STEP_DEG = 0.5;
constexpr int N_THETA = 360.0 / 0.5;
constexpr double OK_FACTOR = 1.20;

constexpr int RING = SAMPLE_RATE; // 1 second ring buffer per channel
constexpr int SEARCH = 4800;      // samples
constexpr float TRIG = 0.02f;

constexpr uint64_t UPDATE_EVERY = 30; // frames (50 * 10ms = 500ms)

static inline double mod360(double deg)
{
    double y = std::fmod(deg, 360.0);
    if (y < 0)
        y += 360.0;
    return y;
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

    PairMeas process(const std::vector<float> &a, const std::vector<float> &b, int i, int j, int maxLag)
    {
        if ((int)a.size() != N_ || (int)b.size() != N_)
            throw std::runtime_error("GCC input size != N");

        for (int k = 0; k < nfft_; k++)
        {
            in1_[k][0] = (k < N_) ? a[k] : 0.0;
            in1_[k][1] = 0.0;
            in2_[k][0] = (k < N_) ? b[k] : 0.0;
            in2_[k][1] = 0.0;
        }
        fftw_execute(p1_);
        fftw_execute(p2_);

        const double eps = 1e-12;
        for (int k = 0; k < nfft_; k++)
        {
            double ar = out1_[k][0], ai = out1_[k][1];
            double br = out2_[k][0], bi = out2_[k][1];
            // (a)*conj(b)
            double re = ar * br + ai * bi;
            double im = ai * br - ar * bi;
            double mag = std::sqrt(re * re + im * im) + eps;
            X_[k][0] = re / mag;
            X_[k][1] = im / mag;
        }

        fftw_execute(pi_);

        maxLag = std::min(maxLag, N_ - 1);
        double pk = -1.0, sumA = 0.0;
        int cnt = 0, bestLag = 0;

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

        // distances
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

        // mic positions
        double dia = std::sqrt(60e-3 * 60e-3 + 35e-3 * 35e-3);
        double R = dia / 2.0;
        double angDeg[M] = {0, 60, 120, 180, 240, 300};
        for (int i = 0; i < M; i++)
        {
            double a = angDeg[i] * PI / 180.0;
            px_[i] = R * std::cos(a);
            py_[i] = R * std::sin(a);
        }

        // precompute pair list + dp + maxLag
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

        // pull last SEARCH samples from ring
        for (int n = 0; n < SEARCH; n++)
        {
            int pos = (writePos - SEARCH + n) % RING;
            if (pos < 0)
                pos += RING;
            for (int ch = 0; ch < M; ch++)
                xSearch_[ch][n] = ring[(size_t)ch * (size_t)RING + (size_t)pos];
        }

        // peak find
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
        if ((float)best < TRIG)
            return false;

        // window around peak (clamped)
        int startK = kPeak - N / 2;
        startK = std::max(0, std::min(startK, SEARCH - N));
        for (int ch = 0; ch < M; ch++)
            std::copy_n(xSearch_[ch].begin() + startK, N, xWin_[ch].begin());

        // DC remove + Hann
        for (int ch = 0; ch < M; ch++)
        {
            double mean = 0.0;
            for (float v : xWin_[ch])
                mean += v;
            mean /= N;
            for (int n = 0; n < N; n++)
                xWin_[ch][n] = (float)((xWin_[ch][n] - mean) * hann_[n]);
        }

        // GCC-PHAT for each pair
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

        // pass 1: best theta
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

        // pass 2: ok mask
        double thresh = errMin * OK_FACTOR;
        for (int t = 0; t < N_THETA; t++)
            ok_[t] = (sseForTheta(t) <= thresh);

        // contiguous ok region around best (circular)
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

        thetaBest = mod360(bestIdx * STEP_DEG - OFFSET);
        return true;
    }

private:
    GCCPhat gcc_;

    std::vector<double> hann_;
    std::array<double, M> px_{};
    std::array<double, M> py_{};

    std::array<PairInfo, 15> pairs_{}; // 6 choose 2
    std::array<double, 15> tau_{};
    std::array<double, 15> w_{};

    std::array<std::vector<float>, M> xSearch_;
    std::array<std::vector<float>, M> xWin_;
    std::vector<uint8_t> ok_;
};

int main()
{
    std::atomic<bool> run{true};
    std::thread stopper([&]
                        {
                            std::string line;
                            if (std::getline(std::cin, line))
                            {
                                run = false; // only stop on real Enter
                            }
                            // if getline fails (EOF), do nothing -> program keeps running
                        });
    if (Pa_Initialize() != paNoError)
        return 1;

    std::cout << "\n\n"; // separate stderr

    PaStream *stream = nullptr;
    PaStreamParameters inParams{};
    inParams.device = DEVICE_INDEX; // <-- your 8ch device index
    inParams.channelCount = IN_CH;
    inParams.sampleFormat = paFloat32;
    inParams.suggestedLatency = Pa_GetDeviceInfo(inParams.device)->defaultHighInputLatency;
    inParams.hostApiSpecificStreamInfo = nullptr;

    PaError err = Pa_OpenStream(&stream,
                                &inParams,
                                nullptr, // output = none
                                SAMPLE_RATE,
                                paFramesPerBufferUnspecified, // <-- avoids ALSA frame-size constraints
                                paNoFlag,
                                nullptr, nullptr);

    if (err != paNoError)
    {
        std::cerr << "Pa_OpenStream failed: " << Pa_GetErrorText(err) << " (" << err << ")\n";
        return 1;
    }

    err = Pa_StartStream(stream);
    if (err != paNoError)
    {
        std::cerr << "Pa_StartStream failed: " << Pa_GetErrorText(err) << " (" << err << ")\n";
        return 1;
    }

    std::array<DenoiseState *, M> st{};
    for (int ch = 0; ch < M; ch++)
    {
        st[ch] = rnnoise_create(nullptr);
        if (!st[ch])
            return 1;
    }

    std::vector<float> ring((size_t)M * (size_t)RING, 0.0f);
    int writePos = 0;
    long long totalWritten = 0;

    std::vector<float> in((size_t)FRAME * (size_t)IN_CH);
    std::vector<float> den((size_t)M * (size_t)FRAME);
    // std::vector<float> play(FRAME);

    std::vector<float> tmp(FRAME);
    std::vector<float> tmp16(FRAME);
    std::vector<float> out16(FRAME);

    DoaSolver solver;

    std::cout << "Denoise ON (RNNoise on channels 0..5) + DOA (GCC-PHAT on denoised). Press Enter to stop.\n";

    uint64_t k = 0;
    while (run)
    {
        PaError e = Pa_ReadStream(stream, in.data(), FRAME);
        if (e == paInputOverflowed)
            continue;
        if (e != paNoError)
        {
            std::cerr << "Pa_ReadStream error: " << Pa_GetErrorText(e) << " (" << e << ")\n";
            break;
        }

        // test ch7 and ch8
        auto rms_ch = [&](int ch)
        {
            double s = 0.0;
            for (int i = 0; i < FRAME; i++)
            {
                float v = in[(size_t)i * (size_t)IN_CH + (size_t)ch];
                s += (double)v * (double)v;
            }
            return std::sqrt(s / FRAME);
        };

        if (k % 50 == 0)
        { // ~ every 0.5s (since FRAME=480 @48k => 10ms per frame)
            std::cout << "RMS ch7=" << rms_ch(6) << "  ch8=" << rms_ch(7) << "\n";
        }

        // RNNoise on ch0..5
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

        // write denoised into ring (time-aligned)
        for (int i = 0; i < FRAME; i++)
        {
            int pos = (writePos + i) % RING;
            for (int ch = 0; ch < M; ch++)
                ring[(size_t)ch * (size_t)RING + (size_t)pos] = den[(size_t)ch * (size_t)FRAME + (size_t)i];
        }
        writePos = (writePos + FRAME) % RING;
        totalWritten += FRAME;

        // play denoised ch0
        // std::copy_n(den.begin(), FRAME, play.begin());
        // Pa_WriteStream(stream, play.data(), FRAME);

        if (++k % UPDATE_EVERY)
            continue;

        double thetaBest = 0.0;
        int Lidx = -1, Ridx = -1;
        if (!solver.compute(ring, writePos, totalWritten, thetaBest, Lidx, Ridx))
            continue;

        std::cout << "DOA = " << thetaBest << " deg";

        if (Lidx >= 0)
        {
            double Lout = mod360(Lidx * STEP_DEG - OFFSET);
            double Rout = mod360(Ridx * STEP_DEG - OFFSET);

            if (Lidx <= Ridx)
                std::cout << "  Range = " << Lout << " .. " << Rout << " deg";
            else
                std::cout << "  Range = " << Lout << " .. 359.5 deg  OR  0 .. " << Rout << " deg";
        }

        std::cout << "\n";
    }

    std::cout << "Stopping...\n";

    for (auto *p : st)
        rnnoise_destroy(p);

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    if (stopper.joinable())
        stopper.join();

    return 0;
}
