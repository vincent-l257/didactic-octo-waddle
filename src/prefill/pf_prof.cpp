#include "pf_prof.hpp"

#include "basalt/log.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace basalt::prefill::detail {
namespace {
struct Dev {
    cudaEvent_t last = nullptr, now = nullptr;
    std::map<std::string, double> ms;
    long long calls = 0;
};
struct Prof {
    bool on = std::getenv("BASALT_PF_PROF") != nullptr;
    int every = on ? std::atoi(std::getenv("BASALT_PF_PROF")) : 0;   // report interval in layers (>1), else the caller's
    std::mutex mu;
    Dev dev[16];
};
Prof& prof() { static Prof p; return p; }
}  // namespace

void pf_tick(cudaStream_t s, const char* label) {
    Prof& p = prof();
    if (!p.on) return;
    int d = 0;
    cudaGetDevice(&d);
    std::lock_guard<std::mutex> lk(p.mu);
    Dev& v = p.dev[d];
    if (!v.last) { cudaEventCreate(&v.last); cudaEventCreate(&v.now); }
    cudaEventRecord(v.now, s);
    cudaEventSynchronize(v.now);
    float f = 0;
    if (label && cudaEventElapsedTime(&f, v.last, v.now) == cudaSuccess) v.ms[label] += f;
    cudaGetLastError();
    std::swap(v.last, v.now);
}

void pf_report(int every) {
    Prof& p = prof();
    if (!p.on) return;
    int d = 0;
    cudaGetDevice(&d);
    std::lock_guard<std::mutex> lk(p.mu);
    Dev& v = p.dev[d];
    if (p.every > 1) every = p.every;
    if (++v.calls % every) return;
    std::string line;
    char b[96];
    for (const auto& [k, ms] : v.ms) {
        std::snprintf(b, sizeof b, " %s %.0f", k.c_str(), ms);
        line += b;
    }
    basalt::log::info("basalt prefill", "pf-prof: CUDA%d %lld layers:%s ms", d, v.calls, line.c_str());
}

}  // namespace basalt::prefill::detail
