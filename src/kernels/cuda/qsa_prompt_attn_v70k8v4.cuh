// src/kernels/cuda/qsa_prompt_attn_v70k8v4.cuh - included by qsa_prompt_attn.cu (its sm_70 cubin holds it).
// sm_70 QSA prompt attention for K8V4 pools (K int8 + fp16 scale per 64, V q4_0 rows of 8 x 18 bytes).
// Same arguments and output as prompt_attn_v70_kernel<3>; one block per (query, KV head), 128 threads, 32-cell chunks.
//   q.k: warp w = cells 8w..8w+7, quad-pair qp = dim group qp (its K scale), FP16 hi + lo q, the K fragment loaded
//        straight from the pool into registers one chunk ahead; the four groups summed by a reduce-scatter.
//   p.v: warp w = dims 64w.. (two q4_0 blocks); A = 2^14 p as FP16 hi + lo (made once per chunk), B = code * w
//        with w the block scale times a power of two, split exactly into FP16 hi + lo (PV_BLO=0: hi only).
//   Row block 1 of both products packs heads 8..11 hi in rows 0..3 and their lo in rows 4..7 (no zero rows).
//   V stays raw q4_0 in shared memory (codes enter exactly); 23 KB of shared memory, 3 blocks per SM.
namespace v70k8v4 {
using namespace basalt::kernels;
constexpr int HD = 256, G = 12, CH = 32, THREADS = 128;   // the mappings below assume these
constexpr int QG = 68;          // halves per q dim group (64 + 4 pad): conflict-free LDS64 across quad-pairs
constexpr int QR = 4 * QG;      // halves per q row
constexpr int VR = 144;         // bytes per V row (8 q4_0 blocks)

#define NEW_MINB 3
#define PV_BLO 1
#define NEW_SINGLE 0

__device__ __forceinline__ void mma884(float* c, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}
// four 4-bit codes in bits 0..3 and 16..19 as the exact half2 (c0 - 8, c1 - 8)
__device__ __forceinline__ uint32_t u4w_to_h2(uint32_t w) {
    uint32_t y = (w & 0x000f000fu) | 0x64006400u;
    __half2 h = *reinterpret_cast<__half2*>(&y);
    h = __hsub2(h, __halves2half2(__float2half(1032.f), __float2half(1032.f)));
    return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t pack_h2(float lo_k, float hi_k) {
    __half2 h = __floats2half2_rn(lo_k, hi_k);
    return *reinterpret_cast<uint32_t*>(&h);
}

struct __align__(16) Smem3 {
    __half qh[G][QR];
    __half ql[G][QR];
    unsigned char v[CH][VR];
    uint2 ks[CH];               // the cell's four K scales (fp16 bits)
    __half wh[8][CH];           // V block scales times a power of two per block (exact)
    float vdown[8];
    float s[G][CH + 4];         // scores
    __half ph[G][CH + 8];       // 2^14 p as hi + lo halves
    __half pl[G][CH + 8];
    long long row[2][CH];
    float qmax[THREADS / 32];
    float alpha[16];
    float lsum[16];
    float mrow[16];
};

__global__ void __launch_bounds__(THREADS, NEW_MINB) attn70_k8v4(const float* __restrict__ q, QsaAttnPools p,
                                                           const int32_t* __restrict__ ids,
                                                           const int32_t* __restrict__ steps, int n_kv_heads,
                                                           int page_size, float scale_log2,
                                                           float* __restrict__ attn, int cap) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem3& S = *reinterpret_cast<Smem3*>(smem_raw);
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int qp = (lane >> 2) & 3, jj = (lane & 3) + 4 * (lane >> 4);
    const int crow = (lane & 1) + 4 * (lane >> 4);
    const int ccol = lane & 2;
    const int nch = (n + CH - 1) / CH;

    auto row_of = [&](int cell) -> long long {
        const long long page = (long long) __ldg(p.page_table + cell / page_size);
        return (page * n_kv_heads + kvh) * page_size + (cell % page_size);
    };
    // chunk 0's and chunk 1's rows
    if (t < CH) {
        S.row[0][t] = t < n ? row_of(__ldg(ids + t)) : -1;
        S.row[1][t] = CH + t < n ? row_of(__ldg(ids + CH + t)) : -1;
    }

    // q: as the baseline (power-of-two scale near 2^14, hi + lo halves), 12 rows
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    for (int i = t; i < G * HD / 4; i += THREADS) {
        const int h = i / (HD / 4), d = (i % (HD / 4)) * 4;
        const float4 x = *reinterpret_cast<const float4*>(q + (size_t) h * HD + d);
        const float xs[4] = {x.x * qup, x.y * qup, x.z * qup, x.w * qup};
        __half hi[4], lo[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) { hi[e] = __float2half_rn(xs[e]); lo[e] = __float2half_rn(xs[e] - __half2float(hi[e])); }
        const int o = (d >> 6) * QG + (d & 63);
        *reinterpret_cast<uint2*>(&S.qh[h][o]) = make_uint2(pack_h2(__half2float(hi[0]), __half2float(hi[1])), pack_h2(__half2float(hi[2]), __half2float(hi[3])));
        *reinterpret_cast<uint2*>(&S.ql[h][o]) = make_uint2(pack_h2(__half2float(lo[0]), __half2float(lo[1])), pack_h2(__half2float(lo[2]), __half2float(lo[3])));
    }
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }
    __syncthreads();   // rows of chunks 0 and 1

    // q.k reads its K fragment straight from the pool: lane (jj, qp) of warp w needs bytes 64qp..64qp+63 of cell
    // 8w + jj's row (four 16-byte loads; a warp instruction covers 8 rows x 64 contiguous bytes)
    uint4 kf[4];
    auto issue_k = [&](int buf) {
        const long long r = S.row[buf][warp * 8 + jj];
        const uint4* src = reinterpret_cast<const uint4*>(p.k_q + (r >= 0 ? r : 0) * HD + qp * 64);
#pragma unroll
        for (int m = 0; m < 4; ++m) kf[m] = r >= 0 ? __ldg(src + m) : make_uint4(0x80808080u, 0x80808080u, 0x80808080u, 0x80808080u);
    };
    // register staging of one chunk: V up to 3 x 16 B, K scales (threads < 32)
    uint4 vx[3];
    uint2 sx = make_uint2(0, 0);
    auto issue = [&](int buf) {
#pragma unroll
        for (int s = 0; s < 3; ++s) {
            const int i = t + s * THREADS;
            vx[s] = make_uint4(0, 0, 0, 0);
            if (i < CH * 9) {
                const int c = i / 9, pc = i % 9;
                const long long r = S.row[buf][c];
                if (r >= 0) vx[s] = __ldg(reinterpret_cast<const uint4*>(p.v_q4 + r * VR) + pc);
            }
        }
        if (t < CH) {
            const long long r = S.row[buf][t];
            sx = r >= 0 ? __ldg(reinterpret_cast<const uint2*>(p.k_scale + r * 4)) : make_uint2(0, 0);
        }
    };
    issue(0);
    issue_k(0);

    float acc[2][2][8];
    float vprev = 1.0f;
#pragma unroll
    for (int rb = 0; rb < 2; ++rb)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[rb][nt][i] = 0.0f;

    for (int ch = 0; ch < nch; ++ch) {
        const int c0 = ch * CH, nh = min(CH, n - c0);
        __syncthreads();   // (A) p.v is done with v, ph/pl, wh; the next chunk's rows are in
#pragma unroll
        for (int s = 0; s < 3; ++s) {
            const int i = t + s * THREADS;
            if (i < CH * 9) *reinterpret_cast<uint4*>(&S.v[i / 9][(i % 9) * 16]) = vx[s];
        }
        if (t < CH) S.ks[t] = sx;
        // the next chunk's loads stay in flight through this chunk's math
        if (ch + 1 < nch) issue((ch + 1) & 1);
        __syncthreads();   // (B)
        // the chunk after next: its ids now, its page-table rows after q.k, stored after p.v
        int nid = 0;
        const bool lookup = t < CH && c0 + 2 * CH + t < n;
        if (lookup) nid = __ldg(ids + c0 + 2 * CH + t);
        // q.k: warp w = cells 8w..8w+7 (B column jj), quad-pair = dim group qp. Row block 0 = heads 0..7 (q hi and
        // lo, separate chains); row block 1 = heads 8..11 hi in rows 0..3 and their lo in rows 4..7 (no zero rows)
        {
            float t0[8], t0l[8], t1[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) t0[i] = t0l[i] = t1[i] = 0.0f;
            const __half* qh0 = &S.qh[jj][qp * QG];
            const __half* ql0 = &S.ql[jj][qp * QG];
            const __half* q1 = jj < 4 ? &S.qh[8 + jj][qp * QG] : &S.ql[4 + jj][qp * QG];
#pragma unroll
            for (int ks = 0; ks < 16; ++ks) {
                const uint4 kq = kf[ks >> 2];
                const uint32_t raw = ((ks & 3) == 0 ? kq.x : (ks & 3) == 1 ? kq.y : (ks & 3) == 2 ? kq.z : kq.w) ^ 0x80808080u;
                // bytes c + 128 into the mantissa of 1024: exact half2 c + 1152, then minus 1152
                const __half2 off = __halves2half2(__float2half(1152.f), __float2half(1152.f));
                uint32_t b0 = __byte_perm(raw, 0x64646464u, 0x5140), b1 = __byte_perm(raw, 0x64646464u, 0x7362);
                {
                    __half2 h0 = __hsub2(*reinterpret_cast<__half2*>(&b0), off), h1 = __hsub2(*reinterpret_cast<__half2*>(&b1), off);
                    b0 = *reinterpret_cast<uint32_t*>(&h0); b1 = *reinterpret_cast<uint32_t*>(&h1);
                }
                const uint2 ah0 = *reinterpret_cast<const uint2*>(qh0 + ks * 4);
                uint2 a1 = *reinterpret_cast<const uint2*>(q1 + ks * 4);
                if (NEW_SINGLE && jj >= 4) a1 = make_uint2(0, 0);
                mma884(t0, ah0.x, ah0.y, b0, b1);
#if !NEW_SINGLE
                const uint2 al0 = *reinterpret_cast<const uint2*>(ql0 + ks * 4);
                mma884(t0l, al0.x, al0.y, b0, b1);
#endif
                mma884(t1, a1.x, a1.y, b0, b1);
            }
            // times the cell's K scale for group qp, summed over the four groups (quad-pairs: lane bits 2, 3)
            float ksc[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int cell = warp * 8 + ccol + 4 * (e >> 1) + (e & 1);
                const uint2 sb = S.ks[cell];
                const uint32_t w = (qp & 2) ? sb.y : sb.x;
                ksc[e] = __half2float(__ushort_as_half((unsigned short) ((qp & 1) ? (w >> 16) : (w & 0xffffu))));
            }
            // reduce-scatter over the quad-pairs: lane bit 2 keeps registers 4 b2.., bit 3 then keeps 4 b2 + 2 b3..
            const bool b2 = qp & 1, b3 = qp >> 1;
            float v4[4], u4[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float ka = ksc[k & 1], kb = ksc[2 | (k & 1)];   // registers k and 4 + k
                const float va = (t0[k] + t0l[k]) * ka, vb = (t0[4 + k] + t0l[4 + k]) * kb;
                const float ua = t1[k] * ka, ub = t1[4 + k] * kb;
                v4[k] = (b2 ? vb : va) + __shfl_xor_sync(0xffffffffu, b2 ? va : vb, 4);
                u4[k] = (b2 ? ub : ua) + __shfl_xor_sync(0xffffffffu, b2 ? ua : ub, 4);
            }
            float v2[2], u2[2];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                v2[j] = (b3 ? v4[2 + j] : v4[j]) + __shfl_xor_sync(0xffffffffu, b3 ? v4[j] : v4[2 + j], 8);
                u2[j] = (b3 ? u4[2 + j] : u4[j]) + __shfl_xor_sync(0xffffffffu, b3 ? u4[j] : u4[2 + j], 8);
                u2[j] += __shfl_xor_sync(0xffffffffu, u2[j], 16);   // + the lo rows of heads 8..11
            }
            // this lane now holds register 4 b2 + 2 b3 + j: row crow + 2 b3, cell 8w + ccol + 4 b2 + j
            {
                const int row = crow + 2 * b3, cell = warp * 8 + ccol + 4 * b2;
                *reinterpret_cast<float2*>(&S.s[row][cell]) = make_float2(v2[0], v2[1]);
                if (lane < 16) *reinterpret_cast<float2*>(&S.s[8 + row][cell]) = make_float2(u2[0], u2[1]);
            }
        }
        if (ch + 1 < nch) issue_k((ch + 1) & 1);   // in flight through the softmax and p.v
        long long nrow = -1;
        if (lookup) {
            const long long page = (long long) __ldg(p.page_table + nid / page_size);
            nrow = (page * n_kv_heads + kvh) * page_size + (nid % page_size);
        }
        __syncthreads();   // (C)
        // online softmax (8 threads per row, 4 cells each), and the V block scales
        {
            constexpr int PER = CH / 8;
            const int r = t >> 3, sub = t & 7;
            if (r < G) {
            float x[PER], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const int c = sub * PER + j;
                x[j] = c < nh ? S.s[r][c] * qdown : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                x[j] = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                sum += x[j];
            }
            {
                __half h[PER], l[PER];
#pragma unroll
                for (int j = 0; j < PER; ++j) {
                    const float y = x[j] * 16384.0f;
                    h[j] = __float2half_rn(y);
                    l[j] = __float2half_rn(y - __half2float(h[j]));
                }
                *reinterpret_cast<uint2*>(&S.ph[r][sub * PER]) = make_uint2(__half_as_ushort(h[0]) | ((uint32_t) __half_as_ushort(h[1]) << 16), __half_as_ushort(h[2]) | ((uint32_t) __half_as_ushort(h[3]) << 16));
                *reinterpret_cast<uint2*>(&S.pl[r][sub * PER]) = make_uint2(__half_as_ushort(l[0]) | ((uint32_t) __half_as_ushort(l[1]) << 16), __half_as_ushort(l[2]) | ((uint32_t) __half_as_ushort(l[3]) << 16));
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
            }
            {   // the V block scales times 2^(11 - e), e the exponent of the chunk's largest per block: exact FP16 w
                const int b = t >> 4, c = (t & 15) * 2;
                const float d0 = __half2float(*reinterpret_cast<const __half*>(&S.v[c][b * 18]));
                const float d1 = __half2float(*reinterpret_cast<const __half*>(&S.v[c + 1][b * 18]));
                float vm = fmaxf(fabsf(d0), fabsf(d1));
#pragma unroll
                for (int o = 1; o < 16; o <<= 1) vm = fmaxf(vm, __shfl_xor_sync(0xffffffffu, vm, o));
                int e = 0;
                if (vm > 0.0f) frexpf(vm, &e);              // vm < 2^e: the scaled scales stay below 2^11
                const float vup = ldexpf(1.0f, 11 - e);
                *reinterpret_cast<__half2*>(&S.wh[b][c]) = __floats2half2_rn(d0 * vup, d1 * vup);
                if ((t & 15) == 0) S.vdown[b] = ldexpf(1.0f, e - 11 - 14);
            }
        }
        __syncthreads();   // (D)
        // p.v: warp w = dims 64w.. = q4_0 blocks 2w, 2w+1; quad-pair qp: block 2w + qp/2, bytes 8(qp & 1)..+7 of
        // it, n-tile nt = the nibble (dims 16nt + 8(qp & 1) + jj of the block). A = p (2^14 p as FP16 hi + lo, made
        // once per chunk); B = code * w split exactly into FP16 hi + lo (w = the block scale times a power of two).
        // Row block 0 = heads 0..7: p_hi B_hi + p_hi B_lo + p_lo B_hi; block 1 = heads 8..11 p_hi (rows 0..3) and
        // p_lo (rows 4..7) times B_hi and B_lo.
        {
            const int blk = 2 * warp + (qp >> 1);
            const float vdown = S.vdown[blk];
            const float rescale = vprev / vdown;   // a power of two: acc is kept in units of this chunk's vdown
            vprev = vdown;
            float tmp[2][2][8];
#pragma unroll
            for (int rb = 0; rb < 2; ++rb)
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                    for (int i = 0; i < 8; ++i) tmp[rb][nt][i] = 0.0f;
            const unsigned char* vb = &S.v[0][blk * 18 + 2 + 8 * (qp & 1) + jj];
            const __half* ph0 = &S.ph[jj][0];
            const __half* pl0 = &S.pl[jj][0];
            const __half* p1 = jj < 4 ? &S.ph[8 + jj][0] : &S.pl[4 + jj][0];
            const __half* wv = &S.wh[blk][0];
#pragma unroll 4
            for (int ks = 0; ks < CH / 4; ++ks) {
                const int cc = ks * 4;
                const uint2 ah = *reinterpret_cast<const uint2*>(ph0 + cc);
                const uint2 al = *reinterpret_cast<const uint2*>(pl0 + cc);
                const uint2 a1 = *reinterpret_cast<const uint2*>(p1 + cc);
                const uint2 wr = *reinterpret_cast<const uint2*>(wv + cc);
                const __half2 w01 = *reinterpret_cast<const __half2*>(&wr.x), w23 = *reinterpret_cast<const __half2*>(&wr.y);
                const uint32_t c0 = vb[(cc + 0) * VR], c1 = vb[(cc + 1) * VR];
                const uint32_t c2 = vb[(cc + 2) * VR], c3 = vb[(cc + 3) * VR];
                const uint32_t x01 = c0 | (c1 << 16), x23 = c2 | (c3 << 16);
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const uint32_t k01 = u4w_to_h2(nt ? (x01 >> 4) : x01), k23 = u4w_to_h2(nt ? (x23 >> 4) : x23);
                    const __half2 bh01 = __hmul2(*reinterpret_cast<const __half2*>(&k01), w01);
                    const __half2 bh23 = __hmul2(*reinterpret_cast<const __half2*>(&k23), w23);
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&bh01), b1 = *reinterpret_cast<const uint32_t*>(&bh23);
                    mma884(tmp[0][nt], ah.x, ah.y, b0, b1);
                    mma884(tmp[1][nt], a1.x, a1.y, b0, b1);
#if !NEW_SINGLE
                    mma884(tmp[0][nt], al.x, al.y, b0, b1);
#endif
#if PV_BLO
                    const __half2 bl01 = __hfma2(*reinterpret_cast<const __half2*>(&k01), w01, __hneg2(bh01));
                    const __half2 bl23 = __hfma2(*reinterpret_cast<const __half2*>(&k23), w23, __hneg2(bh23));
                    const uint32_t e0 = *reinterpret_cast<const uint32_t*>(&bl01), e1 = *reinterpret_cast<const uint32_t*>(&bl23);
                    mma884(tmp[0][nt], ah.x, ah.y, e0, e1);
                    mma884(tmp[1][nt], a1.x, a1.y, e0, e1);
#endif
                }
            }
            // block 1 rows 4..7 (lanes >= 16) are the lo parts of heads 8..11: they take that head alpha
            const float a00 = S.alpha[crow] * rescale, a01 = S.alpha[crow + 2] * rescale;
            const int h1 = 8 + crow - (lane >= 16 ? 4 : 0);
            const float a10 = S.alpha[h1] * rescale, a11 = S.alpha[h1 + 2] * rescale;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[0][nt][i] = fmaf(acc[0][nt][i], ((i >> 1) & 1) ? a01 : a00, tmp[0][nt][i]);
                    acc[1][nt][i] = fmaf(acc[1][nt][i], ((i >> 1) & 1) ? a11 : a10, tmp[1][nt][i]);
                }
        }
        if (t < CH && ch + 2 < nch) S.row[ch & 1][t] = nrow;
    }
    __syncthreads();
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[1][nt][i] += __shfl_xor_sync(0xffffffffu, acc[1][nt][i], 16);
#pragma unroll
    for (int rb = 0; rb < 2; ++rb)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = rb * 8 + crow + 2 * h;
            if (row >= G) continue;
            const float l = S.lsum[row];
            const float inv = l > 0.0f ? vprev / l : 0.0f;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int i = hh * 4 + h * 2;
                    const int d = warp * 64 + (qp >> 1) * 32 + nt * 16 + (qp & 1) * 8 + ccol + 4 * hh;
                    *reinterpret_cast<float2*>(attn + (size_t) row * HD + d) =
                        make_float2(acc[rb][nt][i] * inv, acc[rb][nt][i + 1] * inv);
                }
        }
}
}  // namespace v70k8v4
#undef NEW_MINB
#undef PV_BLO
#undef NEW_SINGLE
