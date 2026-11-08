// src/prefill/gemm.cu - see include/basalt/prefill/gemm.hpp.
#include "basalt/prefill/gemm.hpp"
#include "basalt/kernels/dequant_bf16.hpp"
#include "basalt/kernels/hc_fused.hpp"
#include "basalt/kernels/cc_launch.hpp"
#include "gemm_sm70.cuh"
#include <cuda_bf16.h>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>


#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include "basalt/log.hpp"


namespace basalt::prefill {
namespace {

void ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        basalt::log::error("basalt", "prefill gemm: %s: cuBLAS status %d", what, (int) s);
        std::exit(1);
    }
}

// A setup call whose failure the engine survives (the handle keeps its defaults), but says so.
void note(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) basalt::log::warn("basalt", "prefill gemm: %s: cuBLAS status %d (continuing)", what, (int) s);
}


// Y[t][n] = sum_k X[t][k] W[n][k] (+ beta Y) in FP32 for a cc 7.0 card, which this cuBLAS has no kernels for:
// 64 x 64 tiles, 256 threads of 4 x 4 outputs, K in steps of 16.
template <class T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }

template <class T>
__global__ void __launch_bounds__(256) gemm_simt_kernel(const T* X, const T* W, float* Y, int nt, int nn, int nk,
                                                        int ldy, float beta) {
    __shared__ float xs[16][64 + 1], ws[16][64 + 1];
    const int t0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < nk; k0 += 16) {
        for (int i = threadIdx.x; i < 64 * 16; i += 256) {
            const int r = i / 16, k = i % 16;
            const bool kin = k0 + k < nk;
            xs[k][r] = kin && t0 + r < nt ? to_f(X[(size_t) (t0 + r) * nk + k0 + k]) : 0.0f;
            ws[k][r] = kin && n0 + r < nn ? to_f(W[(size_t) (n0 + r) * nk + k0 + k]) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) { a[i] = xs[k][ty + 16 * i]; b[i] = ws[k][tx + 16 * i]; }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int t = t0 + ty + 16 * i;
        if (t >= nt) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + tx + 16 * j;
            if (n >= nn) continue;
            float* y = Y + (size_t) t * ldy + n;
            *y = beta != 0.0f ? acc[i][j] + beta * *y : acc[i][j];
        }
    }
}

template <class T>
void gemm_simt(const uint16_t* X, const uint16_t* W, float* Y, int64_t T_, int64_t N, int64_t K, int64_t ldy,
               float beta, void* stream) {
    if (sm70_gemm::gemm_sm70_fits(X, W, K)) {   // the tensor cores when the shape allows
        sm70_gemm::gemm_sm70((const T*) X, (const T*) W, Y, (int) T_, (int) N, (int) K, (int) ldy, beta,
                             [&](auto fn, dim3 g, dim3 b, size_t smem, auto... a) {
                                 basalt::kernels::launch_cc(fn, g, b, smem, (cudaStream_t) stream, a...);
                             });
        return;
    }
    const dim3 grid((unsigned) ((N + 63) / 64), (unsigned) ((T_ + 63) / 64));
    basalt::kernels::launch_cc(gemm_simt_kernel<T>, grid, dim3(256), 0, (cudaStream_t) stream, (const T*) X,
                               (const T*) W, Y, (int) T_, (int) N, (int) K, (int) ldy, beta);
}

// One handle made ahead of time per device, taken by the first Gemm that starts there.
cublasHandle_t g_prepared[16] = {};

}  // namespace

bool Gemm::prepare_handle(std::string& err) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16 || g_prepared[dev] != nullptr || basalt::kernels::sm70_current()) return true;
    if (const cublasStatus_t s = cublasCreate(&g_prepared[dev]); s != CUBLAS_STATUS_SUCCESS) {
        g_prepared[dev] = nullptr;
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    return true;
}

Gemm::~Gemm() {
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    if (!external_) {
        if (scratch_) cudaFree(scratch_);
        if (workspace_) cudaFree(workspace_);
    }
}

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    if (basalt::kernels::sm70_current()) {   // no handle: the products run gemm_simt
        stream_ = stream;
        external_ = true;
        workspace_ = workspace;
        scratch_ = scratch;
        scratch_elems_ = scratch_elems;
        return true;
    }
    cublasHandle_t h = nullptr;
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev >= 0 && dev < 16 && g_prepared[dev] != nullptr) {
        h = g_prepared[dev];
        g_prepared[dev] = nullptr;
    } else if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    external_ = true;
    note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    workspace_ = workspace;
    note(cublasSetWorkspace(h, workspace_, ws_bytes), "cublasSetWorkspace");
    note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    return true;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    if (handle_) cublasSetWorkspace((cublasHandle_t) handle_, workspace_, ws_bytes);
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    // every failure names the call and the real status, so "no VRAM" can be told from a broken install
    cublasHandle_t h = nullptr;
    const bool simt = basalt::kernels::sm70_current();
    if (const cublasStatus_t s = simt ? CUBLAS_STATUS_SUCCESS : cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    if (h) note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    // A fixed workspace so the handle never allocates on the way (and graphs could capture it later).
    const size_t ws = 32u << 20;
    if (const cudaError_t e = cudaMalloc(&workspace_, ws); e != cudaSuccess) {
        err = std::string("prefill gemm: workspace of 32 MiB: ") + cudaGetErrorString(e);
        return false;
    }
    if (h) {
        note(cublasSetWorkspace(h, workspace_, ws), "cublasSetWorkspace");
        note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
    }
    if (scratch_elems > 0) {
        if (const cudaError_t e = cudaMalloc((void**) &scratch_, (size_t) scratch_elems * 2); e != cudaSuccess) {
            err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB: " +
                  cudaGetErrorString(e);
            return false;
        }
    }
    scratch_elems_ = scratch_elems;
    return true;
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    if (!handle_) { gemm_simt<__nv_bfloat16>(X, W, Y, T, N, K, ldy, beta, stream_); return; }
    const float alpha = 1.0f;
    // Column-major view: Y^T[N, T] = W[N, K] (stored K x N col-major, transposed) . X^T[K, T].
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16BF, (int) K, X, CUDA_R_16BF, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx");
}

void Gemm::bf16_w(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy) {
    basalt::kernels::Hc8View v;
    if (basalt::kernels::hc8_view(W, v) && v.owned) {
        if (v.rows != N || v.cols != K) {
            basalt::log::error("basalt", "prefill gemm: an 8-bit copy is %d x %d, the product wants %lld x %lld", v.rows,
                         v.cols, (long long) N, (long long) K);
            std::exit(1);
        }
        if (ldy <= 0) ldy = N;
        const int64_t rows = std::min<int64_t>(N, scratch_elems_ / K);
        if (rows <= 0 || scratch_ == nullptr) {
            basalt::log::error("basalt", "prefill gemm: no scratch for an 8-bit matrix (K=%lld)", (long long) K);
            std::exit(1);
        }
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = std::min<int64_t>(rows, N - r0);
            basalt::kernels::hc8_dequant_bf16(v, r0, n, scratch_, stream_);
            bf16(X, scratch_, Y + r0, T, n, K, ldy);
        }
        return;
    }
    bf16(X, W, Y, T, N, K, ldy);
}

const uint16_t* Gemm::bf16_rows(const uint16_t* W, int64_t N, int64_t K) {
    basalt::kernels::Hc8View v;
    if (!basalt::kernels::hc8_view(W, v) || !v.owned) return W;
    if (v.rows != N || v.cols != K || scratch_ == nullptr || N * K > scratch_elems_) return nullptr;
    basalt::kernels::hc8_dequant_bf16(v, 0, N, scratch_, stream_);
    return scratch_;
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    if (!handle_) { gemm_simt<__half>(X, W, Y, T, N, K, ldy, beta, stream_); return; }
    const float alpha = 1.0f;
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16");
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta) {
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { basalt::log::error("basalt", "prefill gemm: scratch too small for K=%lld", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            basalt::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    basalt::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

}  // namespace basalt::prefill
