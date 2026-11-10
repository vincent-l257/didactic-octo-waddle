// tests/kernels/q6k_f16_parity.cpp - the fp16 tensor-core Q6_K projection (cc 7.0) against a double reference.
//
//     build/q6k_f16_parity [--bench]
//
// Random Q6_K blocks (normal fp16 scales) and normal activations with sparse 8x outliers, at the model's projection
// shapes.  Per case and T = 2..8:
//   - the relative RMS of q6k_f16 against a double reference over the dequantized weights and the fp32 activations;
//     it fails above 2e-3, or above the q8_1 path's (native_mmvq on native_quantize_q8_1) relative RMS;
//   - every column of a T-column call is bitwise equal to the same column of the 8-column call.
// --bench: native_mmvq (with its q8_1 quantize) and q6k_f16 (with its prep) timed per T as CUDA graphs over weight
// copies that the L2 cannot hold.  Exits 77 on a card that is not cc 7.x.
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/q6k_f16.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Case {
    const char* name;
    int n_in, n_out;
};

const Case CASES[] = {
    {"gdn qkv", 2560, 10240}, {"gdn z", 2560, 6144}, {"qsa q", 2560, 12288}, {"2560x4096", 2560, 4096},
    {"tail", 1024, 4100},
};

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

float half_to_float(uint16_t h) {
    __half_raw r;
    r.x = h;
    return __half2float(__half(r));
}

// the GGUF Q6_K dequantization (ggml's dequantize_row_q6_K) of one 256-value block, in double
void dequant_block(const uint8_t* b, double* y) {
    uint16_t dd;
    std::memcpy(&dd, b + 208, 2);
    const double d = half_to_float(dd);
    for (int h = 0; h < 2; ++h) {
        const uint8_t* ql = b + 64 * h;
        const uint8_t* qh = b + 128 + 32 * h;
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192 + 8 * h);
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            double* o = y + 128 * h;
            o[l] = d * sc[is] * q1;
            o[l + 32] = d * sc[is + 2] * q2;
            o[l + 64] = d * sc[is + 4] * q3;
            o[l + 96] = d * sc[is + 6] * q4;
        }
    }
}

double rel_rms(const std::vector<float>& y, const std::vector<double>& ref, int n) {
    double se = 0, sr = 0;
    for (int i = 0; i < n; ++i) {
        const double d = y[i] - ref[i];
        se += d * d;
        sr += ref[i] * ref[i];
    }
    return std::sqrt(se / sr);
}

template <class F> float time_graph(cudaStream_t s, F f, int n = 100) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    for (int i = 0; i < n; ++i) f(i);
    cudaStreamEndCapture(s, &g);
    cudaGraphInstantiate(&ge, g, 0);
    cudaGraphLaunch(ge, s);
    cudaStreamSynchronize(s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float best = 1e30f;
    for (int r = 0; r < 7; ++r) {
        cudaEventRecord(e0, s);
        cudaGraphLaunch(ge, s);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        best = std::min(best, ms);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    return best * 1000.0f / (float) n;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace basalt::kernels;
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) return 77;
    cudaDeviceProp pr{};
    if (!ck(cudaGetDeviceProperties(&pr, 0), "properties")) return 1;
    if (pr.major != 7) {
        std::printf("q6k_f16_parity: %s is cc %d.%d, not 7.x: skipped\n", pr.name, pr.major, pr.minor);
        return 77;
    }
    cudaStream_t s;
    if (!ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream")) return 1;
    bool ok = true;
    for (const Case& c : CASES) {
        const int bpr = c.n_in / 256;
        const size_t nb = (size_t) c.n_out * bpr, wb = nb * 210;
        std::mt19937 rng(1234u + (unsigned) c.n_out);
        std::vector<uint8_t> hw(wb);
        for (auto& b : hw) b = (uint8_t) rng();
        for (size_t b = 0; b < nb; ++b) {   // a normal fp16 block scale
            const uint16_t d = (uint16_t) (((5u + rng() % 4u) << 10) | (rng() & 0x3ffu));
            std::memcpy(&hw[b * 210 + 208], &d, 2);
        }
        std::vector<float> hx((size_t) 8 * c.n_in);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : hx) v = nd(rng);
        for (size_t i = 0; i < hx.size(); i += 97) hx[i] *= 8.0f;
        std::vector<double> ref((size_t) 8 * c.n_out), wrow(c.n_in);
        for (int r = 0; r < c.n_out; ++r) {
            for (int kb = 0; kb < bpr; ++kb) dequant_block(&hw[((size_t) r * bpr + kb) * 210], &wrow[(size_t) kb * 256]);
            for (int col = 0; col < 8; ++col) {
                double a = 0;
                for (int k = 0; k < c.n_in; ++k) a += wrow[k] * hx[(size_t) col * c.n_in + k];
                ref[(size_t) col * c.n_out + r] = a;
            }
        }
        const int copies = bench ? (int) std::min<size_t>(8, std::max<size_t>(2, (64u << 20) / wb + 1)) : 1;
        std::vector<void*> w(copies, nullptr);
        float *x = nullptr, *y = nullptr, *y8 = nullptr, *yq = nullptr;
        void *xh = nullptr, *xq = nullptr;
        bool alloc = true;
        for (auto& p : w) alloc = alloc && ck(cudaMalloc(&p, wb), "weights") && ck(cudaMemcpy(p, hw.data(), wb, cudaMemcpyHostToDevice), "copy");
        alloc = alloc && ck(cudaMalloc(&x, hx.size() * 4), "x") && ck(cudaMalloc(&xh, q6k_f16_bytes(c.n_in, 8)), "xh") &&
                ck(cudaMalloc(&xq, native_q8_1_bytes(c.n_in, 8)), "xq") &&
                ck(cudaMalloc(&y, (size_t) 8 * c.n_out * 4), "y") && ck(cudaMalloc(&y8, (size_t) 8 * c.n_out * 4), "y8") &&
                ck(cudaMalloc(&yq, (size_t) 8 * c.n_out * 4), "yq");
        if (!alloc) return 1;
        cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
        // a pageable copy can return before it lands; the work below runs on a non-blocking stream
        if (!ck(cudaDeviceSynchronize(), "copies")) return 1;
        if (!q6k_f16_supported(14, c.n_in, c.n_out, 2, w[0])) {
            std::printf("%-10s not supported\n", c.name);
            return 1;
        }
        q6k_f16_prep(x, xh, c.n_in, 8, s);
        q6k_f16(w[0], xh, y8, c.n_in, c.n_out, 8, s);
        std::vector<float> h8((size_t) 8 * c.n_out), ht((size_t) 8 * c.n_out), hq((size_t) 8 * c.n_out);
        if (!ck(cudaStreamSynchronize(s), "8 columns") ||
            !ck(cudaMemcpy(h8.data(), y8, h8.size() * 4, cudaMemcpyDeviceToHost), "y8"))
            return 1;
        for (int T = 2; T <= 8; ++T) {
            q6k_f16_prep(x, xh, c.n_in, T, s);
            q6k_f16(w[0], xh, y, c.n_in, c.n_out, T, s);
            native_quantize_q8_1(x, xq, c.n_in, T, s);
            native_mmvq(14, w[0], xq, yq, c.n_in, c.n_out, T, s);
            if (!ck(cudaStreamSynchronize(s), "run")) return 1;
            cudaMemcpy(ht.data(), y, (size_t) T * c.n_out * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(hq.data(), yq, (size_t) T * c.n_out * 4, cudaMemcpyDeviceToHost);
            const int n = T * c.n_out;
            const double ef = rel_rms(ht, ref, n), eq = rel_rms(hq, ref, n);
            double mx = 0, mr = 0;
            for (int i = 0; i < n; ++i) {
                mx = std::max(mx, std::fabs(ht[i] - ref[i]));
                mr = std::max(mr, std::fabs(ref[i]));
            }
            const bool same = std::memcmp(ht.data(), h8.data(), (size_t) n * 4) == 0;
            const bool pass = same && ef <= 2e-3 && ef < eq;
            ok = ok && pass;
            std::printf("%-10s %5d->%5d T=%d  f16 rel RMS %.2e max %.2e  q8_1 rel RMS %.2e  columns %s  %s", c.name,
                        c.n_in, c.n_out, T, ef, mx / mr, eq, same ? "T-independent" : "DIFFER", pass ? "ok" : "FAIL");
            if (bench) {
                const float tq = time_graph(s, [&](int i) {
                    native_quantize_q8_1(x, xq, c.n_in, T, s);
                    native_mmvq(14, w[i % copies], xq, yq, c.n_in, c.n_out, T, s);
                });
                const float tf = time_graph(s, [&](int i) {
                    q6k_f16_prep(x, xh, c.n_in, T, s);
                    q6k_f16(w[i % copies], xh, y, c.n_in, c.n_out, T, s);
                });
                const float tk = time_graph(s, [&](int i) { q6k_f16(w[i % copies], xh, y, c.n_in, c.n_out, T, s); });
                const float tm = time_graph(s, [&](int i) { native_mmvq(14, w[i % copies], xq, yq, c.n_in, c.n_out, T, s); });
                std::printf("  | mmvq %.2f us (+q8_1 %.2f)  f16 %.2f us (+prep %.2f)  %+.0f%%", tm, tq, tk, tf,
                            100.0 * (tf - tq) / tq);
            }
            std::printf("\n");
        }
        for (void* p : w) cudaFree(p);
        cudaFree(x); cudaFree(xh); cudaFree(xq); cudaFree(y); cudaFree(y8); cudaFree(yq);
    }
    std::printf("q6k_f16_parity: %s\n", ok ? "pass" : "FAIL");
    return ok ? 0 : 1;
}
