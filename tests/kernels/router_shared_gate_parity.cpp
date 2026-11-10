// tests/kernels/router_shared_gate_parity.cpp - the router GEMV that also dots the shared expert's gate input.
//
// bf16_gemv_fp32_mmvf_gate writes the router logits and, from one more block, the shared expert's gate sigmoid; the
// shared expert then runs without its gate kernels (shared_expert_multi_rows) and the combine applies the gate.
// Checked bitwise against the current chain (bf16_gemv_fp32_mmvf_multi, shared_expert_multi with its own gate,
// dec_moe_combine without one), and against a double-precision host reference for the logits and the gate.
#include "basalt/kernels/bf16_bits.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/dec_moe.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/shared_expert.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

using basalt::kernels::bf16_from_f32;
using basalt::kernels::f32_from_bf16;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <typename T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    check(cudaMalloc((void**) &d, h.size() * sizeof(T)), "malloc");
    check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

template <typename T>
std::vector<T> download(const T* d, size_t n) {
    std::vector<T> h(n);
    check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}

// Q8_0 rows: fp16 scale then 32 int8 codes per block
std::vector<uint8_t> q8_0_rows(std::mt19937& rng, long long rows, long long cols) {
    std::vector<uint8_t> b((size_t) (rows * cols / 32 * 34));
    std::uniform_int_distribution<int> q(-127, 127);
    std::uniform_real_distribution<float> s(0.002f, 0.02f);
    for (size_t o = 0; o < b.size(); o += 34) {
        const float d = s(rng);
        uint32_t u;
        std::memcpy(&u, &d, 4);
        const uint16_t h = (uint16_t) (((u >> 16) & 0x8000u) | ((((u >> 23) & 0xffu) - 112u) << 10) | ((u >> 13) & 0x3ffu));
        std::memcpy(&b[o], &h, 2);
        for (int i = 0; i < 32; ++i) b[o + 2 + (size_t) i] = (uint8_t) (int8_t) q(rng);
    }
    return b;
}

}  // namespace

int main() {
    int dev = 0, major = 0, minor = 0;
    check(cudaGetDevice(&dev), "device");
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    const int N = 2560, NE = 512, K = 10, FF = 640;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> hw((size_t) NE * N);
    for (auto& v : hw) v = bf16_from_f32(nd(rng) * 0.02f);
    const uint16_t* dw = upload(hw);
    std::vector<uint8_t> hg = q8_0_rows(rng, FF, N), hu = q8_0_rows(rng, FF, N), hd = q8_0_rows(rng, N, FF);
    basalt::kernels::NativeSharedWeights nw;
    nw.gate_type = nw.up_type = nw.down_type = 8;
    nw.gate_data = upload(hg); nw.up_data = upload(hu); nw.down_data = upload(hd);
    const int MT = 4;
    float *dx, *dlog0, *dlog1, *dgate, *dup, *dg0, *dg1, *dsh0, *dsh1, *dparts, *dwt, *dy0, *dy1;
    void* dq8;
    check(cudaMalloc((void**) &dx, (size_t) MT * N * 4), "x");
    check(cudaMalloc((void**) &dlog0, (size_t) MT * NE * 4), "logits");
    check(cudaMalloc((void**) &dlog1, (size_t) MT * NE * 4), "logits");
    check(cudaMalloc((void**) &dgate, (size_t) MT * FF * 4), "gate");
    check(cudaMalloc((void**) &dup, (size_t) MT * FF * 4), "up");
    check(cudaMalloc((void**) &dg0, 64), "g");
    check(cudaMalloc((void**) &dg1, 64), "g");
    check(cudaMalloc((void**) &dsh0, (size_t) MT * N * 4), "shared");
    check(cudaMalloc((void**) &dsh1, (size_t) MT * N * 4), "shared");
    check(cudaMalloc((void**) &dparts, (size_t) MT * K * N * 4), "parts");
    check(cudaMalloc((void**) &dwt, (size_t) MT * K * 4), "weights");
    check(cudaMalloc((void**) &dy0, (size_t) MT * N * 4), "y");
    check(cudaMalloc((void**) &dy1, (size_t) MT * N * 4), "y");
    check(cudaMalloc(&dq8, (size_t) MT * FF * 2 + (size_t) MT * N * 2), "q8_1");
    basalt::kernels::DmPlan* dplan;
    check(cudaMalloc((void**) &dplan, sizeof(basalt::kernels::DmPlan)), "plan");
    check(cudaMemset(dplan, 0, sizeof(basalt::kernels::DmPlan)), "plan");   // every entry CUDA0's own row
    nw.q8_1 = dq8;
    cudaStream_t s, s2;
    cudaEvent_t join;
    check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    check(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking), "stream");
    check(cudaEventCreateWithFlags(&join, cudaEventDisableTiming), "event");

    // gate rows scaled so the dots span the sigmoid's flat ends, its middle and the exp underflow
    const float gscale[] = {0.0f, 1e-30f, 1e-4f, 0.02f, 0.3f, 3.0f, 30.0f};
    long long diffs = 0, cases = 0;
    double ref_log = 0, ref_g = 0;
    for (int T = 1; T <= MT; ++T)
        for (float gs : gscale)
            for (int rep = 0; rep < 2; ++rep) {
                std::vector<float> hx((size_t) T * N), hp((size_t) T * K * N), hwt((size_t) T * K);
                for (auto& v : hx) v = nd(rng);
                for (auto& v : hp) v = nd(rng);
                for (auto& v : hwt) v = std::fabs(nd(rng)) * 0.1f;
                std::vector<uint16_t> hgi((size_t) N);
                for (auto& v : hgi) v = bf16_from_f32((rep ? -1.0f : 1.0f) * std::fabs(nd(rng)) * gs);
                const uint16_t* dgi = upload(hgi);
                check(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x");
                check(cudaMemcpy(dparts, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice), "parts");
                check(cudaMemcpy(dwt, hwt.data(), hwt.size() * 4, cudaMemcpyHostToDevice), "weights");
                // the current chain
                basalt::kernels::bf16_gemv_fp32_mmvf_multi(dx, N, dw, dlog0, NE, N, NE, T, s);
                basalt::kernels::shared_expert_multi(T, dx, nw, dgi, dgate, dup, dg0, dsh0, N, FF, s, false);
                basalt::kernels::dec_moe_combine(dparts, nullptr, nullptr, dplan, dwt, dsh0, dy0, N, K, T, nullptr,
                                                 nullptr, 0, s);
                // the router's extra row, the shared expert's projections alone, the combine's gate
                basalt::kernels::bf16_gemv_fp32_mmvf_gate(dx, N, dw, dlog1, NE, N, NE, T, dgi, dg1, s);
                basalt::kernels::native_quantize_q8_1(dx, dq8, N, T, s);
                check(cudaEventRecord(join, s), "fork");
                check(cudaStreamWaitEvent(s2, join, 0), "fork");
                basalt::kernels::shared_expert_multi_rows(T, nw, dgate, dup, dsh1, N, FF, s, rep ? nullptr : s2,
                                                          rep ? nullptr : join);   // up beside gate, or after it
                basalt::kernels::dec_moe_combine(dparts, nullptr, nullptr, dplan, dwt, dsh1, dy1, N, K, T, nullptr,
                                                 nullptr, 0, s, false, nullptr, nullptr, nullptr, dg1);
                check(cudaStreamSynchronize(s), "sync");
                const auto l0 = download(dlog0, (size_t) T * NE), l1 = download(dlog1, (size_t) T * NE);
                const auto y0 = download(dy0, (size_t) T * N), y1 = download(dy1, (size_t) T * N);
                const auto g1 = download(dg1, (size_t) T);
                diffs += std::memcmp(l0.data(), l1.data(), l0.size() * 4) != 0;
                for (size_t i = 0; i < y0.size(); ++i) diffs += std::memcmp(&y0[i], &y1[i], 4) != 0;
                // host reference over the same bf16 bits
                for (int t = 0; t < T; ++t) {
                    double gdot = 0;
                    for (int i = 0; i < N; ++i) gdot += (double) hx[(size_t) t * N + i] * f32_from_bf16(hgi[(size_t) i]);
                    const double gr = 1.0 / (1.0 + std::exp(-gdot));
                    ref_g = std::fmax(ref_g, std::fabs(gr - g1[(size_t) t]));
                    for (int o = 0; o < NE; o += 37) {
                        double acc = 0;
                        for (int i = 0; i < N; ++i)
                            acc += (double) hx[(size_t) t * N + i] * f32_from_bf16(hw[(size_t) o * N + i]);
                        ref_log = std::fmax(ref_log, std::fabs(acc - l1[(size_t) t * NE + o]));
                    }
                }
                ++cases;
                cudaFree((void*) dgi);
            }
    check(cudaGetLastError(), "kernels");
    std::printf("sm_%d%d: %lld cases, %lld differences from the current chain; host reference: logits max |d| %.3g, "
                "gate max |d| %.3g\n", major, minor, cases, diffs, ref_log, ref_g);
    const bool ok = diffs == 0 && ref_log < 1e-3 && ref_g < 1e-5;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
