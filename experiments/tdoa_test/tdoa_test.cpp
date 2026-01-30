// tdoa_min.cpp
// Build: g++ -O3 -std=c++17 tdoa_min.cpp -lsndfile -lfftw3 -lm -o tdoa_min
// Run:   ./tdoa_min recording_8ch_nc.wav [startT endT]

#include <sndfile.h>
#include <fftw3.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr double PI = 3.14159265358979323846;

struct Wav
{
    int fs = 0, ch = 0;
    long long frames = 0;
    std::vector<float> x; // interleaved
};

static Wav readWav(const std::string &path)
{
    SF_INFO info{};
    SNDFILE *f = sf_open(path.c_str(), SFM_READ, &info);
    if (!f)
        throw std::runtime_error("sf_open failed");
    Wav w;
    w.fs = info.samplerate;
    w.ch = info.channels;
    w.frames = info.frames;
    w.x.resize((size_t)w.frames * (size_t)w.ch);
    if (sf_readf_float(f, w.x.data(), info.frames) != info.frames)
    {
        sf_close(f);
        throw std::runtime_error("short read");
    }
    sf_close(f);
    return w;
}
static inline float sample(const Wav &w, long long n, int ch0)
{
    return w.x[(size_t)n * (size_t)w.ch + (size_t)ch0];
}

struct PairMeas
{
    int i, j;
    double tau;
    double w;
}; // i,j are 0-based

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

    // Returns: best lag (within +/-maxLag), tau (sec), and peakRatio used as weight.
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
            double r = x_[idx][0] / nfft_; // real part, scaled
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

static inline double mod360(double deg)
{
    double y = std::fmod(deg, 360.0);
    if (y < 0)
        y += 360.0;
    return y;
}

int main(int argc, char **argv)
{
    try
    {

        const int M = 6;             // use ch0..ch5
        const int N = 512;           // frameLen
        const double c = 343.0;      // speed of sound
        const double offset = 150.0; // your calibration: theta = thetaRaw - 150
        const double stepDeg = 0.5;
        const int nTheta = (int)std::round(360.0 / stepDeg); // 720
        const double okFactor = 1.20;                        // within 20% of min error

        std::string file = "recording_8ch_nc.wav";
        double startT = 4.5;
        double endT = 4.6;

        Wav w = readWav(file);
        if (w.ch < M)
            throw std::runtime_error("wav has fewer channels than M");

        long long s0 = (long long)std::floor(startT * w.fs);
        long long s1 = (long long)std::floor(endT * w.fs);
        s0 = std::max(0LL, s0);
        s1 = std::min(w.frames - 1, s1);
        if (s1 <= s0)
            throw std::runtime_error("bad crop range");

        int L = (int)(s1 - s0 + 1);
        // xSearch[ch][n]
        std::vector<std::vector<float>> xSearch(M, std::vector<float>(L));
        for (int n = 0; n < L; n++)
            for (int ch = 0; ch < M; ch++)
                xSearch[ch][n] = sample(w, s0 + n, ch);

        // find clap peak in chunk: env(n)=max_ch abs
        int kPeak = 0;
        double best = -1;
        for (int n = 0; n < L; n++)
        {
            double env = 0;
            for (int ch = 0; ch < M; ch++)
                env = std::max(env, (double)std::abs(xSearch[ch][n]));
            if (env > best)
            {
                best = env;
                kPeak = n;
            }
        }

        // extract N samples around kPeak (clamped)
        int startK = std::max(0, kPeak - N / 2);
        int endK = std::min(L - 1, startK + N - 1);
        startK = std::max(0, endK - N + 1);

        std::vector<std::vector<float>> x(M, std::vector<float>(N));
        for (int ch = 0; ch < M; ch++)
            for (int n = 0; n < N; n++)
                x[ch][n] = xSearch[ch][startK + n];

        // DC remove + Hann
        std::vector<double> hann(N);
        for (int n = 0; n < N; n++)
            hann[n] = 0.5 * (1.0 - std::cos(2.0 * PI * n / (N - 1)));
        for (int ch = 0; ch < M; ch++)
        {
            double mean = 0;
            for (float v : x[ch])
                mean += v;
            mean /= N;
            for (int n = 0; n < N; n++)
                x[ch][n] = (float)((x[ch][n] - mean) * hann[n]);
        }

        // distance matrix (meters), 0-based [i][j]
        double d[6][6] = {};
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

        // mic positions: regular hex from your formula
        double dia = std::sqrt(60e-3 * 60e-3 + 35e-3 * 35e-3);
        double R = dia / 2.0;
        double angDeg[6] = {0, 60, 120, 180, 240, 300};
        double px[6], py[6];
        for (int i = 0; i < 6; i++)
        {
            double a = angDeg[i] * PI / 180.0;
            px[i] = R * std::cos(a);
            py[i] = R * std::sin(a);
        }

        // GCC-PHAT per pair => tau + peakRatio weights
        GCCPhat gcc(N, (double)w.fs);
        std::vector<PairMeas> meas;
        meas.reserve(M * (M - 1) / 2);

        double wMax = 1e-12;
        for (int i = 0; i < M; i++)
        {
            for (int j = i + 1; j < M; j++)
            {
                int maxLag = (int)std::ceil((d[i][j] / c) * w.fs);
                auto m = gcc.process(x[i], x[j], i, j, maxLag);
                wMax = std::max(wMax, m.w);
                meas.push_back(m);
            }
        }
        for (auto &m : meas)
            m.w /= wMax;

        // precompute dp = pi - pj
        struct DP
        {
            double x, y;
        };
        std::vector<DP> dp(meas.size());
        for (size_t k = 0; k < meas.size(); k++)
        {
            int i = meas[k].i, j = meas[k].j;
            dp[k] = {px[i] - px[j], py[i] - py[j]};
        }

        // pass 1: find best theta (min weighted SSE)
        int bestIdx = 0;
        double errMin = 1e300;
        for (int t = 0; t < nTheta; t++)
        {
            double th = (t * stepDeg) * PI / 180.0;
            double ux = std::cos(th), uy = std::sin(th);
            double sse = 0;
            for (size_t k = 0; k < meas.size(); k++)
            {
                double tauPred = (dp[k].x * ux + dp[k].y * uy) / c;
                double e = meas[k].tau - tauPred;
                double we = meas[k].w * e;
                sse += we * we;
            }
            if (sse < errMin)
            {
                errMin = sse;
                bestIdx = t;
            }
        }

        // pass 2: ok mask + contiguous region around best (circular)
        double thresh = errMin * okFactor;
        std::vector<uint8_t> ok(nTheta, 0);
        for (int t = 0; t < nTheta; t++)
        {
            double th = (t * stepDeg) * PI / 180.0;
            double ux = std::cos(th), uy = std::sin(th);
            double sse = 0;
            for (size_t k = 0; k < meas.size(); k++)
            {
                double tauPred = (dp[k].x * ux + dp[k].y * uy) / c;
                double e = meas[k].tau - tauPred;
                double we = meas[k].w * e;
                sse += we * we;
            }
            ok[t] = (sse <= thresh);
        }

        int Lidx = bestIdx, Ridx = bestIdx;
        if (ok[bestIdx])
        {
            while (true)
            {
                int nl = (Lidx - 1 + nTheta) % nTheta;
                if (nl == bestIdx || !ok[nl])
                    break;
                Lidx = nl;
            }
            while (true)
            {
                int nr = (Ridx + 1) % nTheta;
                if (nr == bestIdx || !ok[nr])
                    break;
                Ridx = nr;
            }
        }
        else
        {
            Lidx = Ridx = -1;
        }

        double thetaRaw = bestIdx * stepDeg;
        double thetaBest = mod360(thetaRaw - offset);

        std::cout << std::fixed;
        std::cout << "DOA = " << thetaBest << " deg\n";
        if (Lidx >= 0)
        {
            double Ldeg = Lidx * stepDeg;
            double Rdeg = Ridx * stepDeg;
            // apply same offset to the range endpoints
            double Lout = mod360(Ldeg - offset);
            double Rout = mod360(Rdeg - offset);

            if (Lidx <= Ridx)
            {
                std::cout << "Range (<=20% min err) = " << Lout << " .. " << Rout << " deg\n";
            }
            else
            {
                // wrap-around in index space -> two intervals
                std::cout << "Range (<=20% min err) = " << Lout << " .. 359.5 deg  OR  0 .. " << Rout << " deg\n";
            }
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
