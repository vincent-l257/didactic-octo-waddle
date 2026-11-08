// include/basalt/kernels/cc_launch.hpp - launches that also reach a compute capability 7.0 card.
//
// The engine's fatbin holds sm_86 and sm_120 code. A cc 7.0 card runs the same device functions from cubins built
// separately for sm_70 (CUDA 12.8, the last toolkit that targets it), loaded with the driver API from the `sm70`
// directory beside the executable. Every launch and every attribute or occupancy query on a kernel goes through the
// functions here: on a cc 7.0 device they resolve the kernel in those cubins by its mangled name; anywhere else they
// are the runtime call itself.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace basalt::kernels {

/// The current device has compute capability 7.0 (cached per device).
bool sm70_current() noexcept;
/// With BASALT_SM70_HOT=1: when the sm_70 cubins have changed on disk since the last call, load them again (new
/// launches use the new code; captured graphs keep the old).  Call between requests.
void sm70_hot_reload();

/// Launch `fn`'s sm_70 image on the current device. `args` points at each parameter in the kernel's own types.
cudaError_t sm70_launch(const void* fn, dim3 grid, dim3 block, size_t smem, cudaStream_t stream, void** args,
                        bool cooperative);

cudaError_t sm70_func_set_attribute(const void* fn, cudaFuncAttribute attr, int value);
cudaError_t sm70_func_get_attributes(cudaFuncAttributes* out, const void* fn);
cudaError_t sm70_occupancy(int* blocks, const void* fn, int block_size, size_t smem);

template <class T> inline cudaError_t cc_func_set_attribute(T* fn, cudaFuncAttribute attr, int value) {
    if (sm70_current()) return sm70_func_set_attribute((const void*) fn, attr, value);
    return cudaFuncSetAttribute((const void*) fn, attr, value);
}
template <class T> inline cudaError_t cc_func_get_attributes(cudaFuncAttributes* out, T* fn) {
    if (sm70_current()) return sm70_func_get_attributes(out, (const void*) fn);
    return cudaFuncGetAttributes(out, (const void*) fn);
}
template <class T>
inline cudaError_t cc_occupancy(int* blocks, T* fn, int block_size, size_t smem) {
    if (sm70_current()) return sm70_occupancy(blocks, (const void*) fn, block_size, smem);
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks, (const void*) fn, block_size, smem);
}

namespace detail {
/// Parameter I of type K: the Ith argument converted, or K{} past the last one (a kernel's default arguments here
/// are all zero or null).
template <size_t I, class K, class In> inline K param(In& in) {
    if constexpr (I < std::tuple_size_v<In>) return K(std::get<I>(in));
    else return K{};
}
template <class... K, class In, size_t... I> inline auto params(In& in, std::index_sequence<I...>) {
    return std::tuple<K...>(param<I, K>(in)...);
}
template <class Tuple, size_t... I>
inline cudaError_t sm70_launch_tuple(const void* fn, dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                                     Tuple& t, bool coop, std::index_sequence<I...>) {
    void* argv[sizeof...(I) + 1] = {(void*) &std::get<I>(t)..., nullptr};
    return sm70_launch(fn, grid, block, smem, stream, argv, coop);
}
}  // namespace detail

/// The kernel's own parameter types: a value passed as another type is converted first, so the bytes the sm_70
/// launch hands over match the kernel's signature.
template <typename... KArgs, typename... Args>
inline cudaError_t sm70_launch_as(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                                  bool coop, Args&&... args) {
    static_assert(sizeof...(Args) <= sizeof...(KArgs), "more arguments than the kernel takes");
    auto in = std::forward_as_tuple(std::forward<Args>(args)...);
    auto t = detail::params<std::remove_cv_t<KArgs>...>(in, std::index_sequence_for<KArgs...>{});
    return detail::sm70_launch_tuple((const void*) kernel, grid, block, smem, stream, t, coop,
                                     std::index_sequence_for<KArgs...>{});
}

/// cudaLaunchKernelEx. On cc 7.0 the attributes are dropped except the cooperative one: the callers that ask for a
/// programmatic edge or a cluster pick kernels that need neither there.
template <typename... KArgs, typename... Args>
inline cudaError_t cc_launch_ex(const cudaLaunchConfig_t* cfg, void (*kernel)(KArgs...), Args&&... args) {
    if (sm70_current()) {
        bool coop = false;
        for (unsigned i = 0; i < cfg->numAttrs; ++i)
            if (cfg->attrs[i].id == cudaLaunchAttributeCooperative && cfg->attrs[i].val.cooperative) coop = true;
        return sm70_launch_as(kernel, cfg->gridDim, cfg->blockDim, cfg->dynamicSmemBytes, cfg->stream, coop,
                              std::forward<Args>(args)...);
    }
    return cudaLaunchKernelEx(cfg, kernel, std::forward<Args>(args)...);
}

/// cudaLaunchCooperativeKernel.
template <class T>
inline cudaError_t cc_launch_cooperative(T* kernel, dim3 grid, dim3 block, void** args, size_t smem,
                                         cudaStream_t stream) {
    if (sm70_current()) return sm70_launch((const void*) kernel, grid, block, smem, stream, args, true);
    return cudaLaunchCooperativeKernel((const void*) kernel, grid, block, args, smem, stream);
}

#if defined(__CUDACC__)
/// `kernel<<<grid, block, smem, stream>>>(args...)`, or its sm_70 image on a cc 7.0 card. A failed sm_70 launch is
/// logged and fatal: the runtime's last-error state does not see driver launches.
template <typename... KArgs, typename... Args>
inline void launch_cc(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                      Args&&... args) {
    if (sm70_current()) {
        (void) sm70_launch_as(kernel, grid, block, smem, stream, false, std::forward<Args>(args)...);
        return;
    }
    if constexpr (sizeof...(Args) == sizeof...(KArgs)) {
        kernel<<<grid, block, smem, stream>>>(std::forward<Args>(args)...);
    } else {   // a pointer carries no default arguments: the trailing ones are passed as their zero defaults
        auto in = std::forward_as_tuple(std::forward<Args>(args)...);
        auto t = detail::params<std::remove_cv_t<KArgs>...>(in, std::index_sequence_for<KArgs...>{});
        std::apply([&](auto&... p) { kernel<<<grid, block, smem, stream>>>(p...); }, t);
    }
}
#endif

}  // namespace basalt::kernels
