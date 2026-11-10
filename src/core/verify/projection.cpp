#include "common.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/kernels/dense_tc.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/q6k_f16.hpp"
#include <cmath>
#include <mutex>
#include <set>
#include <string>
#include <tuple>

namespace basalt::core::verify_detail {

// one stderr line per distinct fast-path fallback (what, a, b)
void note_fallback(const char* what, int a, int b) {
    static std::mutex mu;
    static std::set<std::tuple<std::string, int, int>> seen;
    std::lock_guard<std::mutex> lk(mu);
    if (seen.insert({what, a, b}).second)
        basalt::log::warn("basalt verify", "fallback: %s (types %d / %d)", what, a, b);
}
// a dense projection of the window: dense_tc where dense_tc_mask has the projection's bit and the type, else
// native_mmvq (either one for every window width, so a column's value never depends on T)
void dense_proj(int bit, int type, const void* w, const void* xq, float* y, int n_in, int n_out, int n, void* s) {
    dense_proj(bit, type, w, xq, nullptr, y, n_in, n_out, n, s);
}
// with xh: Q6_K on fp16 tensor cores on a cc 7.0 card for every window width (q6k_f16.hpp)
void dense_proj(int bit, int type, const void* w, const void* xq, const void* xh, float* y, int n_in, int n_out, int n,
                void* s) {
    if ((kernels::dense_tc_mask() & bit) && kernels::dense_tc_supported(type, n_in, n_out, w) &&
        kernels::dense_tc_type_on(type))
        kernels::dense_tc(type, w, xq, y, n_in, n_out, n, s);
    else if (xh && kernels::q6k_f16_supported(type, n_in, n_out, n, w))
        kernels::q6k_f16(w, xh, y, n_in, n_out, n, s);
    else {
        if (kernels::dense_tc_mask() & bit) note_fallback("dense_tc projection on native_mmvq", bit, type);
        kernels::native_mmvq(type, w, xq, y, n_in, n_out, n, s);
    }
}

const void* dense_f16_rows(const float* x, void* xh, int n_in, int n, std::initializer_list<DenseOut> outs, void* s) {
    for (const DenseOut& o : outs)
        if (kernels::q6k_f16_supported(o.type, n_in, o.n_out, n, o.w)) {
            kernels::q6k_f16_prep(x, xh, n_in, n, s);
            return xh;
        }
    return nullptr;
}

// Memory bandwidth of a card from its attributes, bytes per second.
double card_bandwidth(int dev) {
    int khz = 0, bits = 0;
    if (cudaDeviceGetAttribute(&khz, cudaDevAttrMemoryClockRate, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&bits, cudaDevAttrGlobalMemoryBusWidth, dev) != cudaSuccess) {
        cudaGetLastError();
        return 0.0;
    }
    return 2.0 * khz * 1e3 * bits / 8.0;
}

}  // namespace basalt::core::verify_detail
