// SPDX-License-Identifier: MIT
// dp4a forms of the MMQ dot products for cards without int8 tensor cores.  They read the same shared-memory tiles
// and accumulate into the same per-thread slots as the tensor-core forms (the 16 x 8 C-tile layout), so loading,
// write-back and the stream-k fixup are shared.  Each thread holds its rows' x values in registers and walks its
// columns, so every shared-memory load feeds several dp4a.
#pragma once

namespace mmq_dp4a {

static __device__ __forceinline__ void load8(int (&v)[8], const int * p) {
    const int4 a = ((const int4 *) p)[0], b = ((const int4 *) p)[1];
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

template <int N>
static __device__ __forceinline__ int dot(const int * a, const int * b) {
    int s = 0;
#pragma unroll
    for (int k = 0; k < N; ++k) s = ggml_cuda_dp4a(a[k], b[k], s);
    return s;
}

// Q8_0-layout x (one float scale per 32 values) against q8_1 y (D4 or DS4 scales).
template <int sram_stride, int ntx, int J, mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void q8_0_q8_1(const int * x_qs, const float * x_df, const int * y_qs,
                                                 const float * y_df, const half2 * y_ds, float * sum, int i0, int k00) {
    const int li = threadIdx.x / 4, lj = (threadIdx.x % 4) * 2;
#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
        const int k0 = k00 + k01;
        int xv[ntx][2][8];
        float xd[ntx][2];
#pragma unroll
        for (int n = 0; n < ntx; ++n)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int i = i0 + n * 16 + h * 8 + li;
                load8(xv[n][h], x_qs + i * sram_stride + k0);
                xd[n][h] = x_df[i * sram_stride + k0 / QI8_0];
            }
#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx * 8)
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int j = j0 + lj + c;
                int yv[8];
                load8(yv, y_qs + j * MMQ_TILE_Y_K + k01);
                const float dB = ds_layout == MMQ_Q8_1_DS_LAYOUT_D4 ? y_df[j * MMQ_TILE_Y_K + k01 / QI8_1]
                                                                    : __low2float(y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1]);
#pragma unroll
                for (int n = 0; n < ntx; ++n)
#pragma unroll
                    for (int h = 0; h < 2; ++h)
                        sum[(j0 / 8 + n) * 4 + h * 2 + c] += dot<8>(xv[n][h], yv) * xd[n][h] * dB;
            }
    }
}

// Q8_1-layout x (scale and minimum per 32 values) against DS4 y: d_x*d_y*dot + m_x*s_y.
template <int sram_stride, int ntx, int J>
static __device__ __forceinline__ void q8_1_q8_1(const int * x_qs, const half2 * x_dm, const int * y_qs,
                                                 const half2 * y_dm, float * sum, int i0, int k00) {
    const int li = threadIdx.x / 4, lj = (threadIdx.x % 4) * 2;
#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
        const int k0 = k00 + k01;
        int xv[ntx][2][8];
        float2 xd[ntx][2];
#pragma unroll
        for (int n = 0; n < ntx; ++n)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int i = i0 + n * 16 + h * 8 + li;
                load8(xv[n][h], x_qs + i * sram_stride + k0);
                xd[n][h] = __half22float2(x_dm[i * sram_stride + k0 / QI8_1]);
            }
#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx * 8)
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int j = j0 + lj + c;
                int yv[8];
                load8(yv, y_qs + j * MMQ_TILE_Y_K + k01);
                const float2 dsB = __half22float2(y_dm[j * MMQ_TILE_Y_K + k01 / QI8_1]);
#pragma unroll
                for (int n = 0; n < ntx; ++n)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        float & s = sum[(j0 / 8 + n) * 4 + h * 2 + c];
                        s += xd[n][h].x * dsB.x * dot<8>(xv[n][h], yv);
                        s += xd[n][h].y * dsB.y;
                    }
            }
    }
}

// Q3_K-layout x (one float scale per 16 values) against D4 y.
template <int sram_stride, int ntx, int J>
static __device__ __forceinline__ void q8_0_16_q8_1(const int * x_qs, const float * x_df, const int * y_qs,
                                                    const float * y_df, float * sum, int i0, int k00) {
    const int li = threadIdx.x / 4, lj = (threadIdx.x % 4) * 2;
#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
        const int k0 = k00 + k01;
        int xv[ntx][2][8];
        float xd[ntx][2][2];
#pragma unroll
        for (int n = 0; n < ntx; ++n)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int i = i0 + n * 16 + h * 8 + li;
                load8(xv[n][h], x_qs + i * sram_stride + k0);
                xd[n][h][0] = x_df[i * sram_stride + k0 / 4 + 0];
                xd[n][h][1] = x_df[i * sram_stride + k0 / 4 + 1];
            }
#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx * 8)
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int j = j0 + lj + c;
                int yv[8];
                load8(yv, y_qs + j * MMQ_TILE_Y_K + k01);
                const float dB = y_df[j * MMQ_TILE_Y_K + k01 / QI8_1];
#pragma unroll
                for (int n = 0; n < ntx; ++n)
#pragma unroll
                    for (int h = 0; h < 2; ++h)
                        sum[(j0 / 8 + n) * 4 + h * 2 + c] +=
                            dB * (dot<4>(xv[n][h], yv) * xd[n][h][0] + dot<4>(xv[n][h] + 4, yv + 4) * xd[n][h][1]);
            }
    }
}

}  // namespace mmq_dp4a
