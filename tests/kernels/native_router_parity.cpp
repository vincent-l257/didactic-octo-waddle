// tests/kernels/native_router_parity.cpp - the verify window's router (2560 -> 512 BF16 GEMV, top 10 of 512) against
// the one-token kernels and a CPU reference.
//
//     native_router_parity [--bench]
//
// GEMV: bf16_gemv_fp32_mmvf_multi for 1..8 rows (n_in 2560: the fixed-length kernel up to 4 rows; n_in 2048: the
// generic one) must equal bf16_gemv_fp32_mmvf on each row bit for bit, and a double-precision dot to 1e-5 of the sum of
// |w x|. Top 10: native_router_top10_multi for 1..8 rows must equal native_router_top10 on each row bit for bit, on
// Gaussian logits, the GEMV's logits and tie-heavy logits (few distinct values, all equal). Against the CPU rule (the
// ten largest probabilities, ties to the lower expert, weights renormalised over the ten) the ids must match exactly,
// skipping rows where two distinct logits within the top eleven are closer than 1e-5, and the weights to 1e-5.
//
// --bench: the router's two launches at 1..4 rows, cold weights (eight copies in rotation), us per call.
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/native_router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace basalt::kernels;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::printf("native_router_parity: %s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
uint16_t bf16(float f) { uint32_t i; std::memcpy(&i, &f, 4); return (uint16_t) ((i + ((i >> 16) & 1u) + 0x7FFFu) >> 16); }
float from_bf16(uint16_t b) { const uint32_t i = (uint32_t) b << 16; float f; std::memcpy(&f, &i, 4); return f; }
constexpr int NE = 512, K = 10, MAXT = 8;
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd < 1) {
        cudaGetLastError();
        std::printf("native_router_parity: no CUDA device, skipped\n");
        return 77;
    }
    std::mt19937 rng(20261110);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    const int NMAX = 2560;
    uint16_t* d_w;
    float *d_x, *d_y, *d_y1, *d_wt, *d_wt1;
    int32_t *d_id, *d_id1;
    ck(cudaMalloc(&d_w, (size_t) NE * NMAX * 2), "w");
    ck(cudaMalloc(&d_x, (size_t) MAXT * NMAX * 4), "x");
    ck(cudaMalloc(&d_y, (size_t) MAXT * NE * 4), "y");
    ck(cudaMalloc(&d_y1, (size_t) MAXT * NE * 4), "y1");
    ck(cudaMalloc(&d_wt, MAXT * K * 4), "wt");
    ck(cudaMalloc(&d_wt1, MAXT * K * 4), "wt1");
    ck(cudaMalloc(&d_id, MAXT * K * 4), "id");
    ck(cudaMalloc(&d_id1, MAXT * K * 4), "id1");
    long bad_gemv = 0, bad_ref = 0, gemv_rows = 0;
    std::vector<float> gemv_logits;   // the 2560 GEMV's rows, for the top-10 checks
    for (int n_in : {2560, 2048}) {
        for (int T = 1; T <= MAXT; ++T) {
            std::vector<uint16_t> w((size_t) NE * n_in);
            std::vector<float> x((size_t) T * n_in);
            for (auto& v : w) v = bf16(nrm(rng) * 0.05f);
            for (auto& v : x) v = nrm(rng);
            ck(cudaMemcpy(d_w, w.data(), w.size() * 2, cudaMemcpyHostToDevice), "up w");
            ck(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "up x");
            bf16_gemv_fp32_mmvf_multi(d_x, n_in, d_w, d_y, NE, n_in, NE, T, s);
            for (int t = 0; t < T; ++t) bf16_gemv_fp32_mmvf(d_x + (size_t) t * n_in, d_w, d_y1 + (size_t) t * NE, n_in, NE, s);
            ck(cudaStreamSynchronize(s), "gemv");
            std::vector<float> y((size_t) T * NE), y1((size_t) T * NE);
            ck(cudaMemcpy(y.data(), d_y, y.size() * 4, cudaMemcpyDeviceToHost), "y");
            ck(cudaMemcpy(y1.data(), d_y1, y1.size() * 4, cudaMemcpyDeviceToHost), "y1");
            bad_gemv += std::memcmp(y.data(), y1.data(), y.size() * 4) != 0;
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < NE; ++j) {
                    double ref = 0, mag = 0;
                    for (int i = 0; i < n_in; ++i) {
                        const double p = (double) from_bf16(w[(size_t) j * n_in + i]) * x[(size_t) t * n_in + i];
                        ref += p;
                        mag += std::fabs(p);
                    }
                    bad_ref += std::fabs(y[(size_t) t * NE + j] - ref) > 1e-5 * mag;
                }
            gemv_rows += T;
            if (n_in == 2560) gemv_logits.insert(gemv_logits.end(), y.begin(), y.end());
        }
    }
    std::printf("native_router_parity: gemv %ld rows, %ld windows differ from the one-row kernel, %ld outputs off the "
                "CPU dot\n", gemv_rows, bad_gemv, bad_ref);
    long bad_top = 0, bad_cpu = 0, rows = 0, cpu_rows = 0;
    for (int it = 0; it < 4000; ++it) {
        const int T = 1 + it % MAXT, kind = (it / MAXT) % 6;
        std::vector<float> lg((size_t) T * NE);
        for (int t = 0; t < T; ++t) {
            float* l = lg.data() + (size_t) t * NE;
            for (int e = 0; e < NE; ++e) {
                switch (kind) {
                case 0: l[e] = nrm(rng) * 2.0f; break;
                case 1: l[e] = gemv_logits[(size_t) (rng() % (gemv_logits.size() / NE)) * NE + e]; break;
                case 2: l[e] = (float) (int) (nrm(rng) * 3.0f); break;
                case 3: l[e] = (float) (int) (nrm(rng) * 0.6f); break;
                case 4: l[e] = 0.25f; break;
                default: l[e] = nrm(rng) * 8.0f; break;
                }
            }
        }
        ck(cudaMemcpy(d_y, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice), "up logits");
        ck(cudaMemset(d_id, 0xff, MAXT * K * 4), "clear");
        native_router_top10_multi(d_y, d_id, d_wt, T, s);
        for (int t = 0; t < T; ++t) native_router_top10(d_y + (size_t) t * NE, d_id1 + t * K, d_wt1 + t * K, s);
        ck(cudaStreamSynchronize(s), "top10");
        std::vector<int32_t> id(T * K), id1(T * K);
        std::vector<float> wt(T * K), wt1(T * K);
        ck(cudaMemcpy(id.data(), d_id, id.size() * 4, cudaMemcpyDeviceToHost), "id");
        ck(cudaMemcpy(id1.data(), d_id1, id1.size() * 4, cudaMemcpyDeviceToHost), "id1");
        ck(cudaMemcpy(wt.data(), d_wt, wt.size() * 4, cudaMemcpyDeviceToHost), "wt");
        ck(cudaMemcpy(wt1.data(), d_wt1, wt1.size() * 4, cudaMemcpyDeviceToHost), "wt1");
        bad_top += std::memcmp(id.data(), id1.data(), id.size() * 4) != 0 || std::memcmp(wt.data(), wt1.data(), wt.size() * 4) != 0;
        rows += T;
        for (int t = 0; t < T; ++t) {
            const float* l = lg.data() + (size_t) t * NE;
            std::vector<int> ord(NE);
            std::iota(ord.begin(), ord.end(), 0);
            std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return l[a] > l[b]; });
            bool near = false;
            for (int j = 0; j < K; ++j)
                for (int q = j + 1; q <= K; ++q) {
                    const float a = l[ord[j]], b = l[ord[q]];
                    near |= a != b && std::fabs(a - b) < 1e-5f * std::max(1.0f, std::fabs(a));
                }
            if (near) continue;
            const double mx = l[ord[0]];
            double sum = 0, top = 0;
            for (int e = 0; e < NE; ++e) sum += std::exp((double) l[e] - mx);
            for (int j = 0; j < K; ++j) top += std::exp((double) l[ord[j]] - mx) / sum;
            top = std::max(top, 6.103515625e-5);
            for (int j = 0; j < K; ++j) {
                bad_cpu += id[t * K + j] != ord[j];
                bad_cpu += std::fabs(wt[t * K + j] - std::exp((double) l[ord[j]] - mx) / sum / top) > 1e-5;
            }
            ++cpu_rows;
        }
    }
    std::printf("native_router_parity: top10 %ld rows, %ld windows differ from the one-token kernel, %ld entries off "
                "the CPU rule (%ld rows checked)\n", rows, bad_top, bad_cpu, cpu_rows);
    if (bench) {
        cudaDeviceProp pr;
        ck(cudaGetDeviceProperties(&pr, 0), "props");
        uint16_t* wv[8];
        std::vector<uint16_t> w((size_t) NE * NMAX);
        for (auto& v : w) v = bf16(nrm(rng) * 0.05f);
        for (auto& p : wv) {
            ck(cudaMalloc(&p, w.size() * 2), "bench w");
            ck(cudaMemcpy(p, w.data(), w.size() * 2, cudaMemcpyHostToDevice), "bench up");
        }
        auto time = [&](auto&& f) {
            constexpr int kRuns = 100;
            cudaGraph_t g;
            cudaGraphExec_t ge;
            ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
            for (int i = 0; i < kRuns; ++i) f(i);
            ck(cudaStreamEndCapture(s, &g), "end capture");
            ck(cudaGraphInstantiate(&ge, g, 0), "instantiate");
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            float best = 1e30f;
            for (int r = 0; r < 6; ++r) {
                cudaEventRecord(a, s);
                cudaGraphLaunch(ge, s);
                cudaEventRecord(b, s);
                cudaEventSynchronize(b);
                float ms = 0;
                cudaEventElapsedTime(&ms, a, b);
                if (r > 0) best = std::min(best, ms);
            }
            cudaGraphExecDestroy(ge);
            cudaGraphDestroy(g);
            return 1000.0f * best / kRuns;
        };
        for (int T = 1; T <= 4; ++T) {
            const float tg = time([&](int i) { bf16_gemv_fp32_mmvf_multi(d_x, NMAX, wv[i % 8], d_y, NE, NMAX, NE, T, s); });
            const float tt = time([&](int) { native_router_top10_multi(d_y, d_id, d_wt, T, s); });
            const float tb = time([&](int i) {
                bf16_gemv_fp32_mmvf_multi(d_x, NMAX, wv[i % 8], d_y, NE, NMAX, NE, T, s);
                native_router_top10_multi(d_y, d_id, d_wt, T, s);
            });
            const double bytes = (double) NE * NMAX * 2 + (double) T * NMAX * 4;
            std::printf("native_router_parity: bench %s T=%d: gemv 2560->512 %.2f us (%.0f GB/s), top10 %.2f us, "
                        "both %.2f us per window\n", pr.name, T, tg, bytes / (tg * 1e3), tt, tb);
        }
    }
    return bad_gemv == 0 && bad_ref == 0 && bad_top == 0 && bad_cpu == 0 && cpu_rows > 0 ? 0 : 1;
}
