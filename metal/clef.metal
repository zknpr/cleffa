// Clef prefill kernels. Packed varlen batches: tokens of several records are
// concatenated; seq_start[t] is the index of the first token of t's record and
// pos[t] = t - seq_start[t]. Nothing crosses a record boundary.
//
// Precision policy (see README "Accuracy"): the target is the FP32 reference. Weights are the
// exact BF16 values; every matmul accumulates in f32 and takes its activations as FP16 (~8x
// finer than the BF16 the HF reference feeds its Linears), recomputing a record with BF16
// activations if one leaves FP16's range (act16). Everything else (residual stream, norms,
// RoPE, attention, the delta-rule state, the head) is f32.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// ---------------------------------------------------------------- GEMM
// Y[T,N] (+)= X[T,K] . W[N,K]^T   (W bf16; X bf16, half or float; Y f32). MPP masks ragged edges, so any
// T, N, K is valid (verified by bench/edge_test.m), in both multiply and accumulate mode.
struct gemm_args { int T, N, K; };

// Tile shape: see gemm() in clef_metal.m (one shape for all T, for batch invariance).
// XT is the activation type: bfloat, half or float against BF16 weights, all listed by MPP
// (MPPTensorOpsMatMul2d.h). bench/mixed_bench.m: half runs at the bfloat rate with ~8x less
// error; float is exact to f32 accumulation but ~4x slower (used only for the small head GEMMs).
template <typename XT, bool ACC, int TM, int TN>
kernel void gemm_x(constant gemm_args &a [[buffer(0)]],
                   device XT *X [[buffer(1)]],
                   device bfloat *W [[buffer(2)]],
                   device float *Y [[buffer(3)]],
                   uint2 tg [[threadgroup_position_in_grid]]) {
    auto tX = tensor<device XT, dextents<int32_t, 2>, tensor_inline>(X, dextents<int32_t, 2>(a.K, a.T));
    auto tW = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(W, dextents<int32_t, 2>(a.K, a.N));
    auto tY = tensor<device float, dextents<int32_t, 2>, tensor_inline>(Y, dextents<int32_t, 2>(a.N, a.T));
    constexpr auto mode = ACC ? matmul2d_descriptor::mode::multiply_accumulate : matmul2d_descriptor::mode::multiply;
    matmul2d<matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false, mode),
             execution_simdgroups<4>> mm;
    auto mX = tX.slice(0, (int)tg.y * TM);
    auto mW = tW.slice(0, (int)tg.x * TN);
    auto mY = tY.slice((int)tg.x * TN, (int)tg.y * TM);
    mm.run(mX, mW, mY);
}
#define GEMM_VARIANT(XN, XT, TM, TN) \
template [[host_name("gemm_" #XN "_" #TM "x" #TN)]] kernel void gemm_x<XT, false, TM, TN>(constant gemm_args &, device XT *, device bfloat *, device float *, uint2); \
template [[host_name("gemm_" #XN "_acc_" #TM "x" #TN)]] kernel void gemm_x<XT, true, TM, TN>(constant gemm_args &, device XT *, device bfloat *, device float *, uint2);
GEMM_VARIANT(bf16, bfloat, 32, 128)
GEMM_VARIANT(f16, half, 32, 128)
GEMM_VARIANT(f32, float, 32, 128)

// 16-bit GEMM operand: BF16 (the reference's activation type) or FP16 (~8x finer, same GEMM
// rate). Producers store the bits; the GEMM reads them as the matching type.
// FP16 tops out at 65504 (BF16 has float's range). A value past `lim` (65504 unless a test lowers
// it), or a NaN, flags the token's record (slot seq_start[t] of ovf) and the host reruns that
// record alone with BF16 activations; this pass is discarded for it. The value written must
// still be finite: attention multiplies other records' rows by exact zeros (the diagonal rescale,
// P.V over masked keys), and 0 * inf = NaN reached records that never overflowed (review #3,
// tests/test_f16_overflow.sh). So a flagged value saturates to +-65504, which keeps the whole
// discarded pass finite and its products with other records exact zeros. A lowered test limit
// takes this same write path. Whether a record overflows depends only on its own rows, so batch
// invariance holds.
struct act_args { int f16; float lim; };
static inline ushort act16(float v, constant act_args &ac, device atomic_int *ovf) {
    if (!ac.f16) return as_type<ushort>((bfloat)v);
    if (!(fabs(v) <= ac.lim)) {   // also true for NaN
        atomic_store_explicit(ovf, 1, memory_order_relaxed);
        return as_type<ushort>((half)copysign(65504.0f, v));
    }
    return as_type<ushort>((half)v);
}

// ---------------------------------------------------------------- helpers
// Sum over a threadgroup of up to 1024 threads (32 simdgroups).
static inline float tg_sum(float v, threadgroup float *scratch, uint tid, uint nthreads) {
    v = simd_sum(v);
    const uint sg = tid / 32, lane = tid % 32, nsg = (nthreads + 31) / 32;
    if (lane == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? scratch[lane] : 0.0f;
    r = simd_sum(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

static inline float silu(float x) { return x / (1.0f + exp(-x)); }
static inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
// torch.nn.functional.softplus(x), beta=1, threshold=20
// log1p is not in the Metal stdlib; Goldberg's formulation keeps full precision for tiny y.
static inline float log1p_acc(float y) { const float u = 1.0f + y; return u == 1.0f ? y : log(u) * (y / (u - 1.0f)); }
static inline float softplus(float x) { return x > 20.0f ? x : log1p_acc(exp(x)); }

// ---------------------------------------------------------------- embedding
struct embed_args { int H; };
kernel void embed(constant embed_args &a [[buffer(0)]],
                  device const int *ids [[buffer(1)]],
                  device const bfloat *table [[buffer(2)]],
                  device float *x [[buffer(3)]],
                  uint2 gid [[thread_position_in_grid]]) {
    const int t = gid.y, h = gid.x;
    if (h >= a.H) return;
    x[(long)t * a.H + h] = (float)table[(long)ids[t] * a.H + h];
}

// ---------------------------------------------------------------- RMSNorm
// HF Qwen3_5RMSNorm: f32 x * rsqrt(mean(x^2) + eps) * (1 + w); (1 + w) is folded in w.
struct norm_args { int H; float eps; };
kernel void rmsnorm_act(constant norm_args &a [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device const float *w [[buffer(2)]],
                         device ushort *y [[buffer(3)]],
                         constant act_args &ac [[buffer(4)]],
                         device const int *seq_start [[buffer(5)]],
                         device atomic_int *ovf [[buffer(6)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint nt [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    device const float *xr = x + (long)row * a.H;
    float ss = 0.0f;
    for (int i = tid; i < a.H; i += nt) ss += xr[i] * xr[i];
    ss = tg_sum(ss, scratch, tid, nt);
    const float inv = rsqrt(ss / (float)a.H + a.eps);
    device atomic_int *of = ovf + seq_start[row];
    for (int i = tid; i < a.H; i += nt) y[(long)row * a.H + i] = act16(xr[i] * inv * w[i], ac, of);
}

// Same, f32 output (final norm). The head LayerNorm takes it unrounded; HF's BF16 path hands
// the head BF16 here, which CLEF_HEAD_BF16=1 reproduces (layernorm round_input).
kernel void rmsnorm_f32(constant norm_args &a [[buffer(0)]],
                        device const float *x [[buffer(1)]],
                        device const float *w [[buffer(2)]],
                        device float *y [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint nt [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    device const float *xr = x + (long)row * a.H;
    float ss = 0.0f;
    for (int i = tid; i < a.H; i += nt) ss += xr[i] * xr[i];
    ss = tg_sum(ss, scratch, tid, nt);
    const float inv = rsqrt(ss / (float)a.H + a.eps);
    for (int i = tid; i < a.H; i += nt) y[(long)row * a.H + i] = xr[i] * inv * w[i];
}

// ---------------------------------------------------------------- LayerNorm (head)
// torch.nn.LayerNorm(eps=1e-5). Optional BF16 rounding of the input (the reference's
// last_hidden_state is BF16) and both f32 and BF16 outputs.
struct ln_args { int H; float eps; int round_input, write32, write16; };
kernel void layernorm(constant ln_args &a [[buffer(0)]],
                      device const float *x [[buffer(1)]],
                      device const bfloat *w [[buffer(2)]],
                      device const bfloat *b [[buffer(3)]],
                      device float *y32 [[buffer(4)]],
                      device bfloat *y16 [[buffer(5)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint nt [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    device const float *xr = x + (long)row * a.H;
    float s = 0.0f;
    for (int i = tid; i < a.H; i += nt) s += a.round_input ? (float)(bfloat)xr[i] : xr[i];
    const float mean = tg_sum(s, scratch, tid, nt) / (float)a.H;
    float v = 0.0f;
    for (int i = tid; i < a.H; i += nt) {
        const float d = (a.round_input ? (float)(bfloat)xr[i] : xr[i]) - mean;
        v += d * d;
    }
    const float inv = rsqrt(tg_sum(v, scratch, tid, nt) / (float)a.H + a.eps);
    for (int i = tid; i < a.H; i += nt) {
        const float xi = a.round_input ? (float)(bfloat)xr[i] : xr[i];
        const float o = (xi - mean) * inv * (float)w[i] + (float)b[i];
        if (a.write32) y32[(long)row * a.H + i] = o;
        if (a.write16) y16[(long)row * a.H + i] = (bfloat)o;
    }
}

kernel void f32_to_bf16(device const float *x [[buffer(0)]],
                        device bfloat *y [[buffer(1)]],
                        constant long &n [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if ((long)gid < n) y[gid] = (bfloat)x[gid];
}

// ---------------------------------------------------------------- full attention
// QKV row layout (the fused projection): [nh x (query hd | gate hd)] [nkv x hd] [nkv x hd].
// One simdgroup per (token, head); lane l owns dims l, l+32, ..., l+224 (hd = 256), so
// each RoPE pair (i, i+32), i < 32, lives in one lane.
struct attn_prep_args { int nh, nkv, hd, n_rot, row; float eps; };
kernel void attn_prep(constant attn_prep_args &a [[buffer(0)]],
                      device const float *qkv [[buffer(1)]],
                      device const float *qnorm [[buffer(2)]],
                      device const float *knorm [[buffer(3)]],
                      device const int *pos [[buffer(4)]],
                      device const float *inv_freq [[buffer(5)]],
                      device float *Q [[buffer(6)]],    // [nh][T][hd]
                      device float *K [[buffer(7)]],    // [nkv][T][hd]
                      device float *V [[buffer(8)]],    // [nkv][T][hd]
                      device float *G [[buffer(9)]],    // [T][nh*hd]
                      constant int &T [[buffer(10)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const int t = tg.x, head = tg.y;        // head in [0, nh + nkv)
    const int hd = a.hd, per = hd / 32;
    device const float *row = qkv + (long)t * a.row;
    const bool is_q = head < a.nh;
    const int kh = head - a.nh;
    device const float *src = is_q ? row + (long)head * 2 * hd : row + (long)a.nh * 2 * hd + (long)kh * hd;
    float v[8];
    float ss = 0.0f;
    for (int j = 0; j < per; j++) { v[j] = src[lane + 32 * j]; ss += v[j] * v[j]; }
    ss = simd_sum(ss);
    const float inv = rsqrt(ss / (float)hd + a.eps);
    device const float *nw = is_q ? qnorm : knorm;
    for (int j = 0; j < per; j++) v[j] = v[j] * inv * nw[lane + 32 * j];
    // NeoX partial RoPE on the first n_rot dims: rotate_half pairs (i, i + n_rot/2).
    const int rot_half = a.n_rot / 2;   // 32 for Clef: dims lane and lane+32 are slots 0 and 1
    if (lane < (uint)rot_half) {
        const float ang = (float)pos[t] * inv_freq[lane];
        const float c = precise::cos(ang), s = precise::sin(ang);
        const float x0 = v[0], x1 = v[1];
        v[0] = x0 * c - x1 * s;
        v[1] = x1 * c + x0 * s;
    }
    if (is_q) {
        for (int j = 0; j < per; j++) {
            Q[((long)head * T + t) * hd + lane + 32 * j] = v[j];
            G[(long)t * a.nh * hd + (long)head * hd + lane + 32 * j] = src[hd + lane + 32 * j];
        }
    } else {
        device const float *vsrc = row + (long)a.nh * 2 * hd + (long)a.nkv * hd + (long)kh * hd;
        for (int j = 0; j < per; j++) {
            K[((long)kh * T + t) * hd + lane + 32 * j] = v[j];
            V[((long)kh * T + t) * hd + lane + 32 * j] = vsrc[lane + 32 * j];
        }
    }
}

kernel void fill_zero(device float *p [[buffer(0)]],
                      constant long &n [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    if ((long)gid < n) p[gid] = 0.0f;
}

// Causal GQA attention, online softmax in f32. One simdgroup per (query, head).
// Output is multiplied by sigmoid(gate) and written as the 16-bit o_proj operand (act16).
struct attn_args { int nh, nkv, hd, T; float scale; };
kernel void attention(constant attn_args &a [[buffer(0)]],
                      device const float *Q [[buffer(1)]],
                      device const float *K [[buffer(2)]],
                      device const float *V [[buffer(3)]],
                      device const float *G [[buffer(4)]],
                      device const int *seq_start [[buffer(5)]],
                      device ushort *O [[buffer(6)]],   // [T][nh*hd]
                      constant act_args &ac [[buffer(7)]],
                      device atomic_int *ovf [[buffer(8)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const int t = tg.x, h = tg.y, hd = a.hd, per = hd / 32;
    const int kh = h / (a.nh / a.nkv);
    float q[8], acc[8];
    for (int j = 0; j < per; j++) { q[j] = Q[((long)h * a.T + t) * hd + lane + 32 * j] * a.scale; acc[j] = 0.0f; }
    float m = -INFINITY, l = 0.0f;
    device const float *Kh = K + (long)kh * a.T * hd;
    device const float *Vh = V + (long)kh * a.T * hd;
    for (int s = seq_start[t]; s <= t; s++) {
        float d = 0.0f;
        for (int j = 0; j < per; j++) d += q[j] * Kh[(long)s * hd + lane + 32 * j];
        d = simd_sum(d);
        const float mn = max(m, d);
        const float corr = exp(m - mn), p = exp(d - mn);
        l = l * corr + p;
        for (int j = 0; j < per; j++) acc[j] = acc[j] * corr + p * Vh[(long)s * hd + lane + 32 * j];
        m = mn;
    }
    const float invl = 1.0f / l;
    for (int j = 0; j < per; j++) {
        const long o = (long)t * a.nh * hd + (long)h * hd + lane + 32 * j;
        O[o] = act16(acc[j] * invl * sigmoid(G[o]), ac, ovf + seq_start[t]);
    }
}

// Tiled causal GQA attention (FlashAttention-2 style) on f32 simdgroup matrices.
// Threadgroup = one block of 8 queries x one KV head; simdgroup g handles q-head
// kh*G + g (G = nh/nkv), so the G simdgroups share every K/V tile through the cache.
// Keys are processed 32 at a time: S = Q K^T (8x32), online softmax in threadgroup
// scratch, O = diag(alpha) O + P V. Row scaling uses an MMA with a diagonal matrix, which
// does not depend on how simdgroup_matrix elements map to lanes.
// Tile loads may run up to 31 K/V rows (7 Q rows) past T. For every head but the last those
// rows belong to the next head (finite, masked); the last head's tail rows are zeroed on the
// host after attn_prep on every forward, because masked probabilities are exact zeros and
// 0 * NaN would still poison P.V (review #2 M5; tests/test_poison.sh).
constant constexpr int FA_BQ = 8, FA_BK = 32, FA_HD = 256, FA_DT = FA_HD / 8;
kernel void attention_fa(constant attn_args &a [[buffer(0)]],
                         device const float *Q [[buffer(1)]],
                         device const float *K [[buffer(2)]],
                         device const float *V [[buffer(3)]],
                         device const float *G [[buffer(4)]],
                         device const int *seq_start [[buffer(5)]],
                         device ushort *O [[buffer(6)]],
                         constant act_args &ac [[buffer(7)]],
                         device atomic_int *ovf [[buffer(8)]],
                         threadgroup float *smem [[threadgroup(0)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const int i0 = tg.x * FA_BQ, kh = tg.y, grp = a.nh / a.nkv, h = kh * grp + sg;
    const int T = a.T;
    threadgroup float *S = smem + sg * (FA_BQ * FA_BK + 64);   // 8x32 scores/probs
    threadgroup float *D = S + FA_BQ * FA_BK;                   // 8x8 diagonal
    device const float *Qh = Q + ((long)h * T + i0) * FA_HD;
    device const float *Kh = K + (long)kh * T * FA_HD;
    device const float *Vh = V + (long)kh * T * FA_HD;

    simdgroup_float8x8 acc[FA_DT];
    for (int d = 0; d < FA_DT; d++) acc[d] = simdgroup_float8x8(0.0f);
    // softmax state for row r = lane / 4 (replicated over the 4 lanes of the row)
    const int r = lane / 4, c0 = (lane % 4) * 8;
    const int q = i0 + r;
    const bool qvalid = q < T;
    const int qs = qvalid ? seq_start[q] : 0;
    float m = -INFINITY, l = 0.0f;

    // A query block may straddle records. Keys are processed once per record present in
    // the block, in 32-key blocks aligned to that record's start, so every query sees the
    // same blocking (and float summation order) as when its record runs alone: results do
    // not depend on batch composition. Rows of other records are masked in a pass, which
    // is an exact no-op for them (max unchanged -> alpha = 1, probabilities = 0).
    const int k_end = min(i0 + FA_BQ, T);                       // exclusive
    for (int seg = seq_start[i0]; seg >= 0;) {
    for (int kb = seg; kb < k_end; kb += FA_BK) {
        simdgroup_float8x8 s[4];
        for (int j = 0; j < 4; j++) s[j] = simdgroup_float8x8(0.0f);
        for (int d = 0; d < FA_DT; d++) {
            simdgroup_float8x8 qt, kt;
            simdgroup_load(qt, Qh + d * 8, FA_HD);
            for (int j = 0; j < 4; j++) {
                simdgroup_load(kt, Kh + (long)(kb + 8 * j) * FA_HD + d * 8, FA_HD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s[j], qt, kt, s[j]);
            }
        }
        for (int j = 0; j < 4; j++) simdgroup_store(s[j], S + 8 * j, FA_BK);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float v[8], mx = -INFINITY;
        for (int c = 0; c < 8; c++) {
            const int key = kb + c0 + c;
            const bool ok = qvalid && qs == seg && key >= qs && key <= q;
            v[c] = ok ? S[r * FA_BK + c0 + c] * a.scale : -INFINITY;
            mx = max(mx, v[c]);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        const float mn = max(m, mx);
        float alpha = 1.0f, sum = 0.0f;
        if (mn != -INFINITY) {
            alpha = m == -INFINITY ? 0.0f : exp(m - mn);
            for (int c = 0; c < 8; c++) { v[c] = v[c] == -INFINITY ? 0.0f : exp(v[c] - mn); sum += v[c]; }
        } else {
            for (int c = 0; c < 8; c++) v[c] = 0.0f;
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        l = l * alpha + sum;
        m = mn;
        for (int c = 0; c < 8; c++) S[r * FA_BK + c0 + c] = v[c];
        D[lane] = 0.0f;
        D[lane + 32] = 0.0f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane % 4 == 0) D[r * 8 + r] = alpha;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 dm, p[4];
        simdgroup_load(dm, D, 8);
        for (int j = 0; j < 4; j++) simdgroup_load(p[j], S + 8 * j, FA_BK);
        for (int d = 0; d < FA_DT; d++) {
            simdgroup_float8x8 t;
            simdgroup_multiply(t, dm, acc[d]);
            for (int j = 0; j < 4; j++) {
                simdgroup_float8x8 vt;
                simdgroup_load(vt, Vh + (long)(kb + 8 * j) * FA_HD + d * 8, FA_HD);
                simdgroup_multiply_accumulate(t, p[j], vt, t);
            }
            acc[d] = t;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
        int next = -1;   // start of the next record in this query block, if any
        for (int rr = 0; rr < FA_BQ && i0 + rr < T; rr++) {
            if (seq_start[i0 + rr] > seg) { next = seq_start[i0 + rr]; break; }
        }
        seg = next;
    }

    // epilogue: O = diag(1/l) acc, times sigmoid(gate), as the 16-bit o_proj operand
    D[lane] = 0.0f;
    D[lane + 32] = 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane % 4 == 0) D[r * 8 + r] = l > 0.0f ? 1.0f / l : 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dm;
    simdgroup_load(dm, D, 8);
    const int er = lane / 4, ec = (lane % 4) * 2;   // 2 elements per lane in the epilogue
    for (int d = 0; d < FA_DT; d++) {
        simdgroup_float8x8 t;
        simdgroup_multiply(t, dm, acc[d]);
        simdgroup_store(t, S, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const int qq = i0 + er;
        if (qq < T) {
            for (int c = 0; c < 2; c++) {
                const long o = (long)qq * a.nh * FA_HD + (long)h * FA_HD + d * 8 + ec + c;
                O[o] = act16(S[er * 8 + ec + c] * sigmoid(G[o]), ac, ovf + seq_start[qq]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------- Gated DeltaNet
// Projection row layout (fused): [q (Hk*dk) | k (Hk*dk) | v (Hv*dv) | z (Hv*dv) | b (Hv) | a (Hv)].
// Causal depthwise conv (width 4, no bias) + SiLU over the first C = 2*Hk*dk + Hv*dv
// columns, zero left-padding at each record start (nn.Conv1d padding=K-1, first T outputs).
struct conv_args { int C, row, ksize; };
kernel void ssm_conv(constant conv_args &a [[buffer(0)]],
                     device const float *P [[buffer(1)]],
                     device const float *w [[buffer(2)]],      // [C][kernel]
                     device const int *seq_start [[buffer(3)]],
                     device float *out [[buffer(4)]],          // [T][C]
                     uint2 gid [[thread_position_in_grid]]) {
    const int c = gid.x, t = gid.y;
    if (c >= a.C) return;
    const int s0 = seq_start[t];
    float acc = 0.0f;
    for (int j = 0; j < a.ksize; j++) {
        const int tt = t - (a.ksize - 1) + j;
        if (tt >= s0) acc += w[c * a.ksize + j] * P[(long)tt * a.row + c];
    }
    out[(long)t * a.C + c] = silu(acc);
}

// Per (token, head): L2-normalize q and k over dk (eps 1e-6, FLA convention), scale q by
// dk^-1/2; beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias) (ssm_a = -exp(A_log)).
struct gdn_prep_args { int Hk, Hv, dk, dv, row, C; };
kernel void gdn_prep(constant gdn_prep_args &a [[buffer(0)]],
                     device float *X [[buffer(1)]],            // conv output [T][C], q/k normalized in place
                     device const float *P [[buffer(2)]],      // projection rows (for b, a)
                     device const float *ssm_a [[buffer(3)]],
                     device const float *dt_bias [[buffer(4)]],
                     device float *beta [[buffer(5)]],         // [T][Hv]
                     device float *gate [[buffer(6)]],         // [T][Hv] log-decay g
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]]) {
    const int t = tg.x, head = tg.y;   // head < 2*Hk: q/k head; else v-head bookkeeping
    if (head < 2 * a.Hk) {
        device float *x = X + (long)t * a.C + (long)head * a.dk;   // q heads then k heads
        float ss = 0.0f;
        for (int i = lane; i < a.dk; i += 32) ss += x[i] * x[i];
        ss = simd_sum(ss);
        float inv = rsqrt(ss + 1e-6f);
        if (head < a.Hk) inv *= rsqrt((float)a.dk);
        for (int i = lane; i < a.dk; i += 32) x[i] *= inv;
    } else {
        const int h = head - 2 * a.Hk;
        if (lane == 0) {
            const long base = (long)t * a.row + a.C + (long)a.Hv * a.dv;  // skip z (Hv*dv) to b, a
            const float bv = P[base + h], av = P[base + a.Hv + h];
            beta[(long)t * a.Hv + h] = sigmoid(bv);
            gate[(long)t * a.Hv + h] = ssm_a[h] * softplus(av + dt_bias[h]);
        }
    }
}

// Sequential gated delta rule (the recurrent form of torch_recurrent_gated_delta_rule):
//   S *= exp(g_t); kv = S^T k_t; delta = (v_t - kv) * beta_t; S += k_t delta^T; o_t = S^T q_t
// One simdgroup per (v-head, 32/LPC value columns, record); LPC lanes share a column, each
// owning dk/LPC rows of its state in registers. The recurrence is latency-bound, so more
// lanes per column (shorter per-token dependency chain, more simdgroups in flight) is
// faster. The k-head of v-head h is h / (Hv / Hk) (repeat_interleave in the reference).
struct gdn_args { int Hk, Hv, dk, dv, C; };
template <int LPC>
kernel void gdn_scan(constant gdn_args &a [[buffer(0)]],
                     device const float *X [[buffer(1)]],      // [T][C]: q | k | v
                     device const float *beta [[buffer(2)]],
                     device const float *gate [[buffer(3)]],
                     device const int *seq_bounds [[buffer(4)]],  // [n_seq + 1]
                     device float *O [[buffer(5)]],            // [T][Hv*dv]
                     uint3 tg [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]]) {
    constexpr int R = 128 / LPC, CPS = 32 / LPC;   // rows per lane, columns per simdgroup
    const int h = tg.x, col = tg.y * CPS + lane / LPC, part = lane % LPC;
    const int kh = h / (a.Hv / a.Hk);
    const int t0 = seq_bounds[tg.z], t1 = seq_bounds[tg.z + 1];
    const int r0 = part * R;
    float S[R];
    for (int i = 0; i < R; i++) S[i] = 0.0f;
    const long qoff = (long)kh * a.dk + r0;
    const long koff = (long)a.Hk * a.dk + (long)kh * a.dk + r0;
    const long voff = 2L * a.Hk * a.dk + (long)h * a.dv + col;
    for (int t = t0; t < t1; t++) {
        device const float *x = X + (long)t * a.C;
        const float decay = exp(gate[(long)t * a.Hv + h]);
        device const float4 *k4 = (device const float4 *)(x + koff);
        float4 kvp = 0.0f;
        for (int i = 0; i < R / 4; i++) {
            const float4 k = k4[i];
            S[4 * i + 0] *= decay; S[4 * i + 1] *= decay; S[4 * i + 2] *= decay; S[4 * i + 3] *= decay;
            kvp += float4(S[4 * i + 0], S[4 * i + 1], S[4 * i + 2], S[4 * i + 3]) * k;
        }
        float kv = (kvp.x + kvp.y) + (kvp.z + kvp.w);
        for (int off = 1; off < LPC; off <<= 1) kv += simd_shuffle_xor(kv, off);
        const float delta = (x[voff] - kv) * beta[(long)t * a.Hv + h];
        device const float4 *q4 = (device const float4 *)(x + qoff);
        float4 op = 0.0f;
        for (int i = 0; i < R / 4; i++) {
            const float4 k = k4[i], q = q4[i];
            S[4 * i + 0] += k.x * delta; S[4 * i + 1] += k.y * delta;
            S[4 * i + 2] += k.z * delta; S[4 * i + 3] += k.w * delta;
            op += float4(S[4 * i + 0], S[4 * i + 1], S[4 * i + 2], S[4 * i + 3]) * q;
        }
        float o = (op.x + op.y) + (op.z + op.w);
        for (int off = 1; off < LPC; off <<= 1) o += simd_shuffle_xor(o, off);
        if (part == 0) O[(long)t * a.Hv * a.dv + (long)h * a.dv + col] = o;
    }
}
template [[host_name("gdn_scan_2")]] kernel void gdn_scan<2>(constant gdn_args &, device const float *, device const float *, device const float *, device const int *, device float *, uint3, uint);
template [[host_name("gdn_scan_4")]] kernel void gdn_scan<4>(constant gdn_args &, device const float *, device const float *, device const float *, device const int *, device float *, uint3, uint);
template [[host_name("gdn_scan_8")]] kernel void gdn_scan<8>(constant gdn_args &, device const float *, device const float *, device const float *, device const int *, device float *, uint3, uint);

// Qwen3_5RMSNormGated: per (token, v-head) RMSNorm over dv, * w, * silu(z) -> 16-bit out_proj operand.
struct gdn_out_args { int Hv, dv, row, zoff; float eps; };
kernel void gdn_out(constant gdn_out_args &a [[buffer(0)]],
                    device const float *O [[buffer(1)]],
                    device const float *P [[buffer(2)]],
                    device const float *w [[buffer(3)]],
                    device ushort *Y [[buffer(4)]],
                    constant act_args &ac [[buffer(5)]],
                    device const int *seq_start [[buffer(6)]],
                    device atomic_int *ovf [[buffer(7)]],
                    uint2 tg [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]]) {
    const int t = tg.x, h = tg.y;
    device const float *o = O + (long)t * a.Hv * a.dv + (long)h * a.dv;
    device const float *z = P + (long)t * a.row + a.zoff + (long)h * a.dv;
    float ss = 0.0f;
    for (int i = lane; i < a.dv; i += 32) ss += o[i] * o[i];
    ss = simd_sum(ss);
    const float inv = rsqrt(ss / (float)a.dv + a.eps);
    for (int i = lane; i < a.dv; i += 32) {
        Y[(long)t * a.Hv * a.dv + (long)h * a.dv + i] = act16(o[i] * inv * w[i] * silu(z[i]), ac, ovf + seq_start[t]);
    }
}

// ---------------------------------------------------------------- MLP
// [gate | up] projection row -> silu(gate) * up, the 16-bit down_proj operand.
struct swiglu_args { int I; };
kernel void swiglu(constant swiglu_args &a [[buffer(0)]],
                   device const float *GU [[buffer(1)]],
                   device ushort *Y [[buffer(2)]],
                   constant act_args &ac [[buffer(3)]],
                   device const int *seq_start [[buffer(4)]],
                   device atomic_int *ovf [[buffer(5)]],
                   uint2 gid [[thread_position_in_grid]]) {
    const int i = gid.x, t = gid.y;
    if (i >= a.I) return;
    device const float *r = GU + (long)t * 2 * a.I;
    Y[(long)t * a.I + i] = act16(silu(r[i]) * r[a.I + i], ac, ovf + seq_start[t]);
}
