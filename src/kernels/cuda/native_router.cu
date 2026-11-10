// Adapted from topk-moe.cu/common.cuh in llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d; finite F32, 512-expert/10-output path.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "basalt/kernels/native_router.hpp"
#include "basalt/kernels/pdl.hpp"
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include "basalt/kernels/cc_launch.hpp"

namespace basalt::kernels {
namespace {
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
    return value;
}
__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask, 32));
    return value;
}
__launch_bounds__(256, 1)
__global__ void route(const float* __restrict__ logits, int32_t* __restrict__ ids,
                      float* __restrict__ weights) {
    pdl_wait();
    // Preserve the pinned 32x8 block geometry; only row zero is active here.
    // blockIdx.x = the token (a multi-token launch; 0 for the single one)
    logits += (size_t) blockIdx.x * 512; ids += (size_t) blockIdx.x * 10; weights += (size_t) blockIdx.x * 10;
    if (threadIdx.y != 0) return;
    const int lane = threadIdx.x;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    __syncthreads();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            // Deliberately accumulate by WINNING EXPERT lane, not output rank.
            // Multiple selected experts in one lane add in selection order.
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}
// `route` for one token per warp, the same ids and weights bit for bit: the softmax is route's; the selection takes
// the tenth largest lane maximum as a floor (at least ten values reach it, so the ten picks do), ranks every value at
// or above it by the values ordered before it (probability descending, expert ascending), and each lane sums its
// experts' picks in rank order as route's selection loop does.
__launch_bounds__(32)
__global__ void route_window(const float* __restrict__ logits, int32_t* __restrict__ ids,
                             float* __restrict__ weights) {
    __shared__ __align__(16) float s_max[32];
    __shared__ float s_cv[512], s_sv[10];
    __shared__ int32_t s_ci[512], s_si[10];
    pdl_wait();
    logits += (size_t) blockIdx.x * 512; ids += (size_t) blockIdx.x * 10; weights += (size_t) blockIdx.x * 10;
    const int lane = threadIdx.x;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float m8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) m8[i] = fmaxf(values[2 * i], values[2 * i + 1]);
#pragma unroll
    for (int s = 4; s >= 1; s >>= 1)
#pragma unroll
        for (int i = 0; i < s; ++i) m8[i] = fmaxf(m8[2 * i], m8[2 * i + 1]);
    const float lane_max = m8[0];
    s_max[lane] = lane_max;
    __syncwarp();
    int above = 0;   // lane maxima ordered before this one
#pragma unroll
    for (int k = 0; k < 32; k += 4) {
        const float4 o = *reinterpret_cast<const float4*>(s_max + k);
        above += (o.x > lane_max) || (o.x == lane_max && k < lane);
        above += (o.y > lane_max) || (o.y == lane_max && k + 1 < lane);
        above += (o.z > lane_max) || (o.z == lane_max && k + 2 < lane);
        above += (o.w > lane_max) || (o.w == lane_max && k + 3 < lane);
    }
    const float floor_p = __shfl_sync(0xffffffffu, lane_max, __ffs(__ballot_sync(0xffffffffu, above == 9)) - 1, 32);
    unsigned in = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) in |= (unsigned) (values[i] >= floor_p) << i;
    const int mine = __popc(in);
    int end = mine;   // inclusive prefix of the candidate counts
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, end, d, 32);
        if (lane >= d) end += u;
    }
    const int n_cand = __shfl_sync(0xffffffffu, end, 31, 32);
    int at = end - mine;
#pragma unroll
    for (int i = 0; i < 16; ++i)
        if (in >> i & 1u) { s_cv[at] = values[i]; s_ci[at] = lane + i * 32; ++at; }
    __syncwarp();
    for (int j = lane; j < n_cand; j += 32) {
        const float v = s_cv[j];
        const int32_t e = s_ci[j];
        int rank = 0;
        for (int k = 0; k < n_cand; ++k) {
            const float o = s_cv[k];
            rank += (o > v) || (o == v && s_ci[k] < e);
        }
        if (rank < 10) { ids[rank] = e; s_sv[rank] = v; s_si[rank] = e; }
    }
    __syncwarp();
    float selected_sum = 0.0f;
#pragma unroll
    for (int r = 0; r < 10; ++r)
        if ((s_si[r] & 31) == lane) selected_sum += s_sv[r];
    const float selected = lane < 10 ? s_sv[lane] : 0.0f;
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}
// The residency-biased router: the weights are `route`'s, the selection ranks the resident experts' probabilities times
// `boost`. One warp a token.
__launch_bounds__(32, 1)
__global__ void route_prior(const float* __restrict__ logits, ResidentTabs held, float boost, int32_t* __restrict__ ids,
                            float* __restrict__ weights, unsigned long long* counts) {
    pdl_wait();
    logits += (size_t) blockIdx.x * 512; ids += (size_t) blockIdx.x * 10; weights += (size_t) blockIdx.x * 10;
    const int lane = threadIdx.x;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
    float score[16];
    uint32_t resident = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
        bool in = false;
        for (int q = 0; q < held.n; ++q) in |= held.t[q][lane + i * 32] >= 0;
        resident |= (uint32_t) in << i;
        score[i] = in ? values[i] * boost : values[i];
    }
    // the tenth largest probability of the unbiased router, to count the picks the bias changed
    float tenth = 0.0f;
    if (counts != nullptr) {
        float u[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) u[i] = values[i];
        for (int rank = 0; rank < 10; ++rank) {
            float best = u[0];
#pragma unroll
            for (int i = 1; i < 16; ++i) best = max(best, u[i]);
            best = warp_max(best);
            tenth = best;
            int hit = 16;
#pragma unroll
            for (int i = 15; i >= 0; --i) if (u[i] == best) hit = i;
            const unsigned lanes = __ballot_sync(0xffffffffu, hit < 16);
            if (lane == __ffs(lanes) - 1) {
#pragma unroll
                for (int i = 0; i < 16; ++i) if (i == hit) u[i] = -INFINITY;
            }
        }
    }
    float selected = 0.0f, selected_sum = 0.0f;
    int selected_res = 0;
    for (int rank = 0; rank < 10; ++rank) {
        float best = score[0], prob = values[0];
        int expert = lane, in = (int) (resident & 1u);
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (score[i] > best) { best = score[i]; prob = values[i]; expert = lane + i * 32; in = (int) (resident >> i & 1u); }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const float other_prob = __shfl_xor_sync(0xffffffffu, prob, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            const int other_in = __shfl_xor_sync(0xffffffffu, in, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; prob = other_prob; expert = other_id; in = other_in; }
        }
        if ((expert & 31) == lane) {
            score[expert / 32] = -INFINITY;
            ids[rank] = expert;
            selected_sum += prob;
        }
        if (rank == lane) { selected = prob; selected_res = in; }
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
    if (counts != nullptr) {
        const unsigned changed = __ballot_sync(0xffffffffu, lane < 10 && selected < tenth);
        const unsigned outside = __ballot_sync(0xffffffffu, lane < 10 && selected_res == 0);
        if (lane == 0) {
            atomicAdd(&counts[3], 10ull);
            atomicAdd(&counts[4], (unsigned long long) __popc(changed));
            atomicAdd(&counts[5], (unsigned long long) __popc(outside));
        }
    }
}
bool valid(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream) {
    if (!stream || !valid(logits, 512 * 4) || !valid(ids, 10 * 4) || !valid(weights, 10 * 4)
        || overlap(logits, 512 * 4, ids, 10 * 4) || overlap(logits, 512 * 4, weights, 10 * 4)
        || overlap(ids, 10 * 4, weights, 10 * 4))
        throw std::invalid_argument("native router requires a stream, aligned spans, and disjoint outputs");
    basalt::kernels::launch_cc(route, 1, dim3(32, 8), 0, static_cast<cudaStream_t>(stream), logits, ids, weights);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream) {
    if (!stream || n_tok < 1 || !valid(logits, (size_t) n_tok * 512 * 4) || !valid(ids, (size_t) n_tok * 10 * 4) ||
        !valid(weights, (size_t) n_tok * 10 * 4))
        throw std::invalid_argument("native router (multi) requires a stream and aligned [n,512]/[n,10] buffers");
    launch_pdl(route_window, dim3((unsigned) n_tok), dim3(32), 0, static_cast<cudaStream_t>(stream), logits, ids, weights);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void route_prior_top10(const float* logits, const ResidentTabs& held, float lambda, int32_t* ids, float* weights, int n_tok,
                       unsigned long long* counts, void* stream) {
    if (!stream || n_tok < 1 || !valid(logits, (size_t) n_tok * 512 * 4) || !valid(ids, (size_t) n_tok * 10 * 4) ||
        !valid(weights, (size_t) n_tok * 10 * 4))
        throw std::invalid_argument("route prior requires a stream and aligned [n,512]/[n,10] buffers");
    launch_pdl(route_prior, dim3((unsigned) n_tok), dim3(32), 0, static_cast<cudaStream_t>(stream), logits, held,
               std::exp(lambda), ids, weights, counts);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
