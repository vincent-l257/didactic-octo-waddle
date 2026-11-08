// src/core/sm70_cubins.cpp - see include/basalt/kernels/cc_launch.hpp.
//
// The sm_70 cubins come from another toolkit, so their anonymous-namespace names differ from this binary's: a kernel
// is found by its mangled name with every `_GLOBAL__N__<hash>_<file>_<hash>` span reduced to the file part.
#include "basalt/kernels/cc_launch.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/file.hpp"

#include <cuda.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace basalt::kernels {
namespace {

constexpr int kMaxDev = 64;

[[noreturn]] void die(const char* what, CUresult r) {
    const char* name = nullptr;
    cuGetErrorName(r, &name);
    basalt::log::error("basalt sm70", "%s: %s", what, name != nullptr ? name : "?");
    std::abort();
}

/// `_GLOBAL__N__` spans: the digits before it give the span's length; inside it the first `_cu_` ends the file name.
std::string norm_name(const std::string& s) {
    static const std::string key = "_GLOBAL__N__";
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        const size_t p = s.find(key, i);
        if (p == std::string::npos) { out.append(s, i, std::string::npos); break; }
        size_t start = p;
        while (start > i && std::isdigit((unsigned char) s[start - 1])) --start;
        size_t len = 0;
        for (size_t t = start; t < p && len < 100000; ++t) len = len * 10 + (size_t) (s[t] - '0');
        const size_t end = p + len;
        const size_t h = p + key.size();
        size_t d = h + 9, d2 = d;
        while (d2 < end && std::isdigit((unsigned char) s[d2])) ++d2;
        const size_t cu = d2 + 1 < end ? s.find("_cu_", d2 + 1) : std::string::npos;
        const bool ok = start < p && end <= s.size() && h + 8 < end && s[h + 8] == '_' && d2 > d && d2 < end &&
                        s[d2] == '_' && cu != std::string::npos && cu < end;
        if (!ok) { out.append(s, i, h - i); i = h; continue; }
        out.append(s, i, start - i);
        out += "_ANON_";
        out.append(s, d2 + 1, cu - d2 - 1);
        out.push_back('_');
        i = end;
    }
    return out;
}

struct Device {
    std::once_flag loaded;
    CUcontext ctx = nullptr;
    std::vector<CUmodule> mods;
    std::unordered_map<std::string, CUfunction> by_name;
    std::mutex mu;
    std::unordered_map<const void*, CUfunction> by_symbol;
    std::unordered_map<CUfunction, int> dyn_smem;   // the dynamic shared memory each function has opted in to
};

Device g_dev[kMaxDev];

std::filesystem::path cubin_dir() {
    const std::string exe = basalt::platform::module_path();
    return std::filesystem::path(exe).parent_path() / "sm70";
}

void bind(int dev) {
    Device& d = g_dev[dev];
    CUcontext cur = nullptr;
    cuCtxGetCurrent(&cur);
    if (cur != d.ctx) {
        if (const CUresult r = cuCtxSetCurrent(d.ctx); r != CUDA_SUCCESS) die("cuCtxSetCurrent", r);
    }
}

// Load every cubin in the directory into the current context: its modules join d.mods, its kernels go into `names`
// (a kernel two cubins share, an inline template, resolves to the first file's copy).  Returns the files loaded.
int load_dir(Device& d, std::unordered_map<std::string, CUfunction>& names) {
    const std::filesystem::path dir = cubin_dir();
    std::error_code ec;
    int files = 0;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        const std::string name = e.path().filename().string();
        if (!e.is_regular_file() || name.size() < 12 || name.compare(name.size() - 12, 12, ".sm_70.cubin") != 0) continue;
        CUmodule m = nullptr;
        if (const CUresult r = cuModuleLoad(&m, e.path().string().c_str()); r != CUDA_SUCCESS) {
            basalt::log::warn("basalt sm70", "cuModuleLoad %s failed", name.c_str());
            continue;
        }
        d.mods.push_back(m);
        ++files;
        unsigned n = 0;
        if (cuModuleGetFunctionCount(&n, m) != CUDA_SUCCESS || n == 0) continue;
        std::vector<CUfunction> fns(n);
        if (cuModuleEnumerateFunctions(fns.data(), n, m) != CUDA_SUCCESS) continue;
        for (CUfunction f : fns) {
            const char* fname = nullptr;
            if (cuFuncGetName(&fname, f) == CUDA_SUCCESS && fname != nullptr) names.emplace(norm_name(fname), f);
        }
    }
    return files;
}

void load(int dev) {
    Device& d = g_dev[dev];
    std::call_once(d.loaded, [&] {
        if (const CUresult r = cuInit(0); r != CUDA_SUCCESS) die("cuInit", r);
        CUdevice cd;
        if (const CUresult r = cuDeviceGet(&cd, dev); r != CUDA_SUCCESS) die("cuDeviceGet", r);
        if (const CUresult r = cuDevicePrimaryCtxRetain(&d.ctx, cd); r != CUDA_SUCCESS) die("cuDevicePrimaryCtxRetain", r);
        bind(dev);
        const int files = load_dir(d, d.by_name);
        if (files == 0) {
            basalt::log::error("basalt sm70", "no sm_70 cubins in %s", cubin_dir().string().c_str());
            std::abort();
        }
        basalt::log::info("basalt sm70", "CUDA%d: %d cubins, %zu kernels from %s", dev, files, d.by_name.size(),
                          cubin_dir().string().c_str());
    });
    bind(dev);
}

/// The kernel's mangled name. The runtime names a kernel only on a device its fatbin has code for, so on a cc 7.0
/// device the name is asked on another visible card and the current device is restored.
const char* host_name(const void* fn) {
    static std::mutex mu;
    static std::unordered_map<const void*, const char*> names;
    std::lock_guard<std::mutex> lk(mu);
    if (const auto it = names.find(fn); it != names.end()) return it->second;
    // a failing runtime call inside a stream capture ends the capture, so the cc 7.0 device is never asked
    const char* name = nullptr;
    cudaError_t e = cudaErrorNoKernelImageForDevice;
    int major = 0, here = 0;
    cudaGetDevice(&here);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, here);
    if (major != 7) e = cudaFuncGetName(&name, fn);
    if (e != cudaSuccess || name == nullptr) {
        name = nullptr;
        int cur = 0, n = 0;
        cudaGetDevice(&cur);
        cudaGetDeviceCount(&n);
        for (int d = 0; d < n && name == nullptr; ++d) {
            int major = 0;
            if (d == cur || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, d) != cudaSuccess ||
                major == 7)
                continue;
            if (cudaSetDevice(d) == cudaSuccess && cudaFuncGetName(&name, fn) != cudaSuccess) name = nullptr;
            (void) cudaGetLastError();
        }
        cudaSetDevice(cur);
        if (name == nullptr) {
            basalt::log::error("basalt sm70", "kernel name: %s, and no other visible card has its code",
                               cudaGetErrorString(e));
            return nullptr;
        }
    }
    names.emplace(fn, name);
    return name;
}

int current_device() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMaxDev) {
        basalt::log::error("basalt sm70", "no current device");
        std::abort();
    }
    return dev;
}

/// The sm_70 function for `fn` on `dev`, or nullptr when no cubin holds it.
CUfunction find(int dev, const void* fn) {
    load(dev);
    Device& d = g_dev[dev];
    std::lock_guard<std::mutex> lk(d.mu);
    if (const auto it = d.by_symbol.find(fn); it != d.by_symbol.end()) return it->second;
    const char* host = host_name(fn);
    if (host == nullptr) return nullptr;
    CUfunction f = nullptr;
    if (const auto it = d.by_name.find(norm_name(host)); it != d.by_name.end()) f = it->second;
    for (size_t i = 0; f == nullptr && i < d.mods.size(); ++i)
        if (cuModuleGetFunction(&f, d.mods[i], host) != CUDA_SUCCESS) f = nullptr;
    if (f == nullptr) basalt::log::warn("basalt sm70", "no sm_70 image for %s", host);
    d.by_symbol.emplace(fn, f);
    return f;
}

/// Raise the function's dynamic shared memory cap to `bytes` (cc 7.0 takes up to 96 KiB on opt-in).
CUresult opt_in(int dev, CUfunction f, int bytes) {
    Device& d = g_dev[dev];
    std::lock_guard<std::mutex> lk(d.mu);
    int& have = d.dyn_smem[f];
    if (bytes <= have) return CUDA_SUCCESS;
    const CUresult r = cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes);
    if (r == CUDA_SUCCESS) have = bytes;
    return r;
}

}  // namespace

void sm70_hot_reload() {
    static const bool on = [] {
        const char* e = std::getenv("BASALT_SM70_HOT");
        return e != nullptr && *e != 0 && *e != '0';
    }();
    if (!on) return;
    static std::filesystem::file_time_type seen{};
    static bool have = false;
    std::filesystem::file_time_type newest{};
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(cubin_dir(), ec))
        if (e.is_regular_file()) newest = std::max(newest, e.last_write_time(ec));
    if (!have) { seen = newest; have = true; return; }
    if (newest <= seen) return;
    seen = newest;
    CUcontext prev = nullptr;
    cuCtxGetCurrent(&prev);
    for (int dev = 0; dev < kMaxDev; ++dev) {
        Device& d = g_dev[dev];
        if (d.ctx == nullptr) continue;
        std::lock_guard<std::mutex> lk(d.mu);
        if (cuCtxSetCurrent(d.ctx) != CUDA_SUCCESS) continue;
        // the old modules stay loaded: graphs captured from them keep running their code
        std::unordered_map<std::string, CUfunction> names;
        const int files = load_dir(d, names);
        if (files == 0) continue;
        d.by_name.swap(names);
        d.by_symbol.clear();
        basalt::log::info("basalt sm70", "CUDA%d: reloaded %d cubins, %zu kernels", dev, files, d.by_name.size());
    }
    cuCtxSetCurrent(prev);
}

bool sm70_current() noexcept {
    static std::atomic<signed char> known[kMaxDev] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMaxDev) return false;
    signed char k = known[dev].load(std::memory_order_relaxed);
    if (k == 0) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        k = major == 7 && minor == 0 ? 1 : -1;
        known[dev].store(k, std::memory_order_relaxed);
    }
    return k == 1;
}

cudaError_t sm70_launch(const void* fn, dim3 grid, dim3 block, size_t smem, cudaStream_t stream, void** args,
                        bool cooperative) {
    const int dev = current_device();
    CUfunction f = find(dev, fn);
    if (f == nullptr) {
        const char* host = host_name(fn);
        basalt::log::error("basalt sm70", "launch of %s: no sm_70 image", host != nullptr ? host : "?");
        std::abort();
    }
    if (smem > 48 * 1024) {
        if (const CUresult r = opt_in(dev, f, (int) smem); r != CUDA_SUCCESS) die("dynamic shared memory opt-in", r);
    }
    const CUresult r = cooperative
        ? cuLaunchCooperativeKernel(f, grid.x, grid.y, grid.z, block.x, block.y, block.z, (unsigned) smem,
                                    (CUstream) stream, args)
        : cuLaunchKernel(f, grid.x, grid.y, grid.z, block.x, block.y, block.z, (unsigned) smem, (CUstream) stream,
                         args, nullptr);
    if (r != CUDA_SUCCESS) {
        const char* host = host_name(fn);
        basalt::log::error("basalt sm70", "launch of %s grid %u,%u,%u block %u,%u,%u smem %zu", host != nullptr ? host : "?",
                           grid.x, grid.y, grid.z, block.x, block.y, block.z, smem);
        die("cuLaunchKernel", r);
    }
    return cudaSuccess;
}

cudaError_t sm70_func_set_attribute(const void* fn, cudaFuncAttribute attr, int value) {
    const int dev = current_device();
    CUfunction f = find(dev, fn);
    if (f == nullptr) return cudaErrorInvalidDeviceFunction;
    if (attr == cudaFuncAttributeMaxDynamicSharedMemorySize)
        return opt_in(dev, f, value) == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
    if (attr == cudaFuncAttributePreferredSharedMemoryCarveout)
        return cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, value) == CUDA_SUCCESS
                   ? cudaSuccess : cudaErrorInvalidValue;
    return cudaErrorNotSupported;
}

cudaError_t sm70_func_get_attributes(cudaFuncAttributes* out, const void* fn) {
    const int dev = current_device();
    CUfunction f = find(dev, fn);
    if (f == nullptr) return cudaErrorInvalidDeviceFunction;
    *out = cudaFuncAttributes{};
    auto get = [&](CUfunction_attribute a) { int v = 0; cuFuncGetAttribute(&v, a, f); return v; };
    out->sharedSizeBytes = (size_t) get(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
    out->constSizeBytes = (size_t) get(CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES);
    out->localSizeBytes = (size_t) get(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES);
    out->maxThreadsPerBlock = get(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    out->numRegs = get(CU_FUNC_ATTRIBUTE_NUM_REGS);
    out->ptxVersion = get(CU_FUNC_ATTRIBUTE_PTX_VERSION);
    out->binaryVersion = get(CU_FUNC_ATTRIBUTE_BINARY_VERSION);
    out->maxDynamicSharedSizeBytes = get(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES);
    out->preferredShmemCarveout = get(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT);
    return cudaSuccess;
}

cudaError_t sm70_occupancy(int* blocks, const void* fn, int block_size, size_t smem) {
    const int dev = current_device();
    CUfunction f = find(dev, fn);
    if (f == nullptr) return cudaErrorInvalidDeviceFunction;
    return cuOccupancyMaxActiveBlocksPerMultiprocessor(blocks, f, block_size, smem) == CUDA_SUCCESS
               ? cudaSuccess : cudaErrorInvalidValue;
}

}  // namespace basalt::kernels
