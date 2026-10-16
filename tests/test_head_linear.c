/* Packed CPU head weights must preserve the original BLAS results, including the
 * single-row path whose reduction changes when its weight matrix is transposed. */
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <dispatch/dispatch.h>
static size_t dispatched_parts, dispatch_calls;
static void test_dispatch_apply(size_t parts, dispatch_queue_t queue, void (^work)(size_t)) {
    dispatch_calls++;
    dispatched_parts = parts;
    dispatch_apply(parts, queue, work);
}
static bool fail_pack_allocation, guard_pack_allocation;
static size_t pack_allocation_calls, guarded_pack_bytes;
static unsigned char *guarded_pack_base;
static void *test_calloc(size_t n, size_t size) {
    pack_allocation_calls++;
    if (fail_pack_allocation) return NULL;
    if (!guard_pack_allocation) return calloc(n, size);
    assert(!guarded_pack_base && (!size || n <= (SIZE_MAX - 256) / size));
    guarded_pack_bytes = n * size;
    guarded_pack_base = calloc(1, guarded_pack_bytes + 256);
    assert(guarded_pack_base);
    memset(guarded_pack_base, 0xa5, 128);
    memset(guarded_pack_base + 128 + guarded_pack_bytes, 0x5a, 128);
    return guarded_pack_base + 128;
}
#define calloc test_calloc
#define dispatch_apply test_dispatch_apply
#include "../clef_head.c"
#undef calloc
#undef dispatch_apply

static float value(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16;
    return (float)(x & 0xffff) / 32768.0f - 1.0f;
}

/* Original full-width BLAS call, independent of the candidate's split helper. */
static void unsplit_lin(const linear *l, const float *X, int n, float *Y) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, l->out, l->in,
                1.0f, X, l->in, l->w, l->in, 0.0f, Y, l->out);
    if (l->b) for (int r = 0; r < n; r++) for (int j = 0; j < l->out; j++) Y[(size_t)r * l->out + j] += l->b[j];
}

static void check(int in, int out, int rows, int workers) {
    linear l = { .out = out, .in = in, .max_workers = workers };
    l.w = malloc((size_t)out * in * sizeof(float));
    l.b = malloc((size_t)out * sizeof(float));
    float *x = malloc((size_t)rows * in * sizeof(float));
    float *want = malloc((size_t)rows * out * sizeof(float));
    float *got = malloc((size_t)rows * out * sizeof(float));
    assert(l.w && l.b && x && want && got);
    for (size_t i = 0; i < (size_t)out * in; i++) {
        float f = value((uint32_t)i + 73);
        uint32_t bits; memcpy(&bits, &f, sizeof(bits));
        l.w[i] = bf16_to_f32((uint16_t)(bits >> 16));
    }
    for (int i = 0; i < out; i++) l.b[i] = value(i + 91);
    for (size_t i = 0; i < (size_t)rows * in; i++) x[i] = value((uint32_t)i) * (i % 37 ? 1 : 8);
    unsplit_lin(&l, x, rows, want);
    dispatched_parts = 0;
    lin(&l, x, rows, got);
    const size_t original_parts = out % 64 == 0 && (size_t)out * in >= lin_split_min() ? 2 : 0;
    assert(dispatched_parts == original_parts);
    assert(memcmp(want, got, (size_t)rows * out * sizeof(float)) == 0);
    char err[128];
    assert(pack_linear(&l, err, sizeof(err)));
    // Padding is outside the logical tensor and must never affect any output.
    if (l.packed) for (int k = 0; k < in; k++) for (int j = out; j < l.packed_ld; j++)
        l.packed[(size_t)k * l.packed_ld + j] = NAN;
    dispatched_parts = 0;
    lin(&l, x, rows, got);
    const size_t expected_parts = original_parts && l.packed && workers == 4 && rows > 1 && out % 128 == 0 ? 4 : original_parts;
    assert(dispatched_parts == expected_parts);
    if (memcmp(want, got, (size_t)rows * out * sizeof(float))) {
        double max_error = 0;
        int nonfinite = 0;
        for (int i = 0; i < rows * out; i++) {
            max_error = fmax(max_error, fabs(want[i] - got[i]));
            nonfinite += !isfinite(got[i]);
        }
        fprintf(stderr, "head layout mismatch: in=%d out=%d rows=%d max_error=%g nonfinite=%d\n",
                in, out, rows, max_error, nonfinite);
    }
    assert(memcmp(want, got, (size_t)rows * out * sizeof(float)) == 0);
    if (rows == 1 && l.packed) {
        for (int k = 0; k < in; k++) for (int j = 0; j < out; j++)
            l.packed[(size_t)k * l.packed_ld + j] = NAN;
        lin(&l, x, rows, got);
        assert(memcmp(want, got, (size_t)out * sizeof(float)) == 0);
    }
    free_linear(&l); free(x); free(want); free(got);
}

/* The epilogue may use only the existing four column groups. Rejected shapes
 * must leave Y untouched so callers can execute the original lin/GELU path. */
static void check_gelu4(int in, int out, int rows, int workers, bool bias) {
    linear l = { .out = out, .in = in, .max_workers = workers };
    l.w = malloc((size_t)out * in * sizeof(float));
    l.b = bias ? malloc((size_t)out * sizeof(float)) : NULL;
    const size_t cells = (size_t)rows * out;
    float *x = malloc((size_t)rows * in * sizeof(float));
    float *want = malloc(cells * sizeof(float));
    float *guarded = malloc((cells + 2) * sizeof(float)), *got = guarded ? guarded + 1 : NULL;
    assert(l.w && (!bias || l.b) && x && want && guarded);
    for (size_t i = 0; i < (size_t)out * in; i++) l.w[i] = value((uint32_t)i + 73) * 0.125f;
    for (int j = 0; bias && j < out; j++) l.b[j] = value(j + 91) * 3;
    for (size_t i = 0; i < (size_t)rows * in; i++) x[i] = value((uint32_t)i) * (i % 37 ? 1 : 8);
    for (int packed = 0; packed < 2; packed++) {
        if (packed) {
            char err[128]; assert(pack_linear(&l, err, sizeof(err)));
            if (l.packed) for (int k = 0; k < in; k++) for (int j = out; j < l.packed_ld; j++)
                l.packed[(size_t)k * l.packed_ld + j] = NAN;
        }
        lin(&l, x, rows, want);
        for (size_t i = 0; i < cells; i++) want[i] = gelu(want[i]);
        guarded[0] = 1234567.0f; guarded[cells + 1] = -1234567.0f;
        for (size_t i = 0; i < cells; i++) got[i] = NAN;
        dispatch_calls = dispatched_parts = 0;
        const bool used = lin_gelu4(&l, x, rows, got);
        const bool eligible = l.packed && workers == 4 && rows > 1 && out % 128 == 0 &&
                              (size_t)out * in >= lin_split_min();
        assert(used == eligible);
        assert(dispatch_calls == (eligible ? 1u : 0u));
        assert(dispatched_parts == (eligible ? 4u : 0u));
        if (!used) {
            for (size_t i = 0; i < cells; i++) assert(isnan(got[i]));
            lin(&l, x, rows, got);
            for (size_t i = 0; i < cells; i++) got[i] = gelu(got[i]);
        }
        assert(guarded[0] == 1234567.0f && guarded[cells + 1] == -1234567.0f);
        assert(memcmp(want, got, cells * sizeof(float)) == 0);
        for (size_t i = 0; i < cells; i++) assert(isfinite(got[i]));
    }
    free_linear(&l); free(x); free(want); free(guarded);
}

/* At x=1, adding half an upward ULP rounds back to one before GELU. Keeping
 * that sum wider until the following multiply changes the result by one ULP.
 * This fixture must distinguish those graphs, independently of random cases. */
static void check_gelu4_bias_rounding(void) {
    enum { IN = 1024, OUT = 1024, ROWS = 2 };
    const float values[] = {1, -1, 0.125f, -0.125f, 4, -4, 8, -8};
    linear l = { .out = OUT, .in = IN, .max_workers = 4 };
    l.w = calloc((size_t)OUT * IN, sizeof(float)); l.b = malloc(OUT * sizeof(float));
    float *x = calloc((size_t)ROWS * IN, sizeof(float));
    float *want = malloc(ROWS * OUT * sizeof(float)), *got = malloc(ROWS * OUT * sizeof(float));
    assert(l.w && l.b && x && want && got);
    for (int r = 0; r < ROWS; r++) x[r * IN] = 1;
    int distinguishable = 0;
    for (int j = 0; j < OUT; j++) {
        const float a = values[j % (sizeof(values) / sizeof(values[0]))];
        l.w[(size_t)j * IN] = a;
        l.b[j] = (nextafterf(a, INFINITY) - a) * 0.5f;
        volatile float rounded = a + l.b[j];
        const float expected = gelu(rounded);
        for (int r = 0; r < ROWS; r++) want[r * OUT + j] = expected;
        const double unrounded = (double)a + (double)l.b[j];
        const float half = (float)(0.5 * unrounded);
        const float arg = (float)(unrounded * (double)0.70710678118654752f);
        const float delayed = half * (1.0f + erff(arg));
        distinguishable += memcmp(&expected, &delayed, sizeof(float)) != 0;
    }
    assert(distinguishable > 0);
    char err[128]; assert(pack_linear(&l, err, sizeof(err)));
    assert(lin_gelu4(&l, x, ROWS, got));
    assert(memcmp(want, got, ROWS * OUT * sizeof(float)) == 0);
    free_linear(&l); free(x); free(want); free(got);
}

/* Float64 reference for the batched scorer's two affine projections and GELU. */
static void check_scorer(int rows) {
    enum { W = 64, IN = 4 * W };
    linear first = { .out = W, .in = IN }, last = { .out = 1, .in = W };
    first.w = malloc(W * IN * sizeof(float)); first.b = malloc(W * sizeof(float));
    last.w = malloc(W * sizeof(float)); last.b = malloc(sizeof(float));
    float *input = malloc((size_t)rows * IN * sizeof(float));
    float *hidden = malloc(((size_t)rows * W + 1) * sizeof(float));
    float *output = malloc(((size_t)rows + 1) * sizeof(float));
    assert(first.w && first.b && last.w && last.b && input && hidden && output);
    for (int j = 0; j < W; j++) {
        first.b[j] = value(j + 81) * 0.125f;
        last.w[j] = value(j + 912) * 0.125f;
        for (int k = 0; k < IN; k++) {
            float v = value(j * IN + k + 93) * 0.125f;
            uint32_t bits; memcpy(&bits, &v, sizeof(bits));
            first.w[j * IN + k] = bf16_to_f32((uint16_t)(bits >> 16));
        }
    }
    last.b[0] = -0.125f;
    for (int i = 0; i < rows * IN; i++) input[i] = value(i + 71);
    char err[128]; assert(pack_linear(&first, err, sizeof(err)));
    for (int k = 0; k < IN; k++) for (int j = W; j < first.packed_ld; j++)
        first.packed[(size_t)k * first.packed_ld + j] = NAN;
    for (int i = 0; i <= rows * W; i++) hidden[i] = NAN;
    for (int i = 0; i <= rows; i++) output[i] = NAN;
    lin(&first, input, rows, hidden);
    for (int i = 0; i < rows * W; i++) hidden[i] = gelu(hidden[i]);
    lin(&last, hidden, rows, output);
    assert(isnan(hidden[rows * W]) && isnan(output[rows]));
    for (int r = 0; r < rows; r++) {
        double reference = last.b[0];
        for (int j = 0; j < W; j++) {
            double h = first.b[j];
            for (int k = 0; k < IN; k++) h += (double)input[r * IN + k] * first.w[j * IN + k];
            reference += last.w[j] * (0.5 * h * (1.0 + erf(h / sqrt(2.0))));
        }
        assert(isfinite(output[r]) && fabs(output[r] - reference) < 2e-6);
    }
    free_linear(&first); free_linear(&last); free(input); free(hidden); free(output);
}

/* Construct and inspect special floats only through integer bytes: an FP operation
 * in the oracle could quiet a signaling NaN and hide a changed copy. */
static void check_pack_bytes(void) {
    const int shape[][2] = {{32,32},{32,96},{96,32},{96,160},{64,192},
                           {1024,1024},{4096,1024},{5120,1024},{1024,4096},{1024,3072}};
    const uint32_t special[] = {0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,
        0x7f800000,0xff800000,0x7fc12345,0xffc54321,0x7f812345,0xff854321};
    for (size_t si=0;si<sizeof(shape)/sizeof(shape[0]);si++) {
        int in=shape[si][0], out=shape[si][1], ld=out+32;
        size_t cells=(size_t)in*out, bytes=cells*4, packed_bytes=(size_t)in*ld*4;
        unsigned char *source=malloc(bytes+256), *saved=malloc(bytes+256);
        unsigned char *bias=malloc((size_t)out*4), *bias_saved=malloc((size_t)out*4);
        unsigned char *want=calloc(1,packed_bytes);
        assert(source && saved && bias && bias_saved && want);
        memset(source,0x39,bytes+256);
        for (size_t i=0;i<cells;i++) {
            uint32_t bits=(uint32_t)(i*2654435761u)^0x3f012345;
            if (i%17==0) bits=special[(i/17)%(sizeof(special)/sizeof(special[0]))];
            memcpy(source+128+i*4,&bits,4);
        }
        for(int j=0;j<out;j++) {uint32_t bits=special[j%(sizeof(special)/sizeof(special[0]))];memcpy(bias+(size_t)j*4,&bits,4);}
        memcpy(saved,source,bytes+256);memcpy(bias_saved,bias,(size_t)out*4);
        /* Independent naive mapping; full allocation equality includes untouched zero padding. */
        for(int j=0;j<out;j++) for(int k=0;k<in;k++)
            memcpy(want+((size_t)k*ld+j)*4,source+128+((size_t)j*in+k)*4,4);
        linear l={.w=(float *)(source+128),.b=(float *)bias,.in=in,.out=out,.max_workers=4};
        guard_pack_allocation=true;size_t before=pack_allocation_calls;char err[128]={0};
        assert(pack_linear(&l,err,sizeof(err)));guard_pack_allocation=false;
        assert(pack_allocation_calls==before+1 && l.packed_ld==ld && guarded_pack_bytes==packed_bytes);
        assert(l.packed==(float *)(guarded_pack_base+128));
        assert(!memcmp(l.packed,want,packed_bytes) && !memcmp(source,saved,bytes+256) && !memcmp(bias,bias_saved,(size_t)out*4));
        for(int i=0;i<128;i++) assert(guarded_pack_base[i]==0xa5 && guarded_pack_base[128+packed_bytes+i]==0x5a);
        free(guarded_pack_base);guarded_pack_base=NULL;
        free(source);free(saved);free(bias);free(bias_saved);free(want);
    }
    const int ragged[][2]={{63,65},{33,64},{64,33},{1025,1280}};
    for(size_t i=0;i<sizeof(ragged)/sizeof(ragged[0]);i++) {
        uint32_t w=0x7f812345,b=0x80000000;
        linear l={.w=(float *)&w,.b=(float *)&b,.in=ragged[i][0],.out=ragged[i][1],.max_workers=2};
        char err[128]={0};size_t before=pack_allocation_calls;
        assert(pack_linear(&l,err,sizeof(err)) && pack_allocation_calls==before);
        assert(l.w==(float *)&w && l.b==(float *)&b && !l.packed && !l.packed_ld && l.max_workers==2);
        assert(w==0x7f812345 && b==0x80000000);
    }
    const int invalid[][2]={{0,32},{-1,32},{32,0},{32,-1},{32,INT_MAX}};
    for(size_t i=0;i<sizeof(invalid)/sizeof(invalid[0]);i++) {
        linear l={.in=invalid[i][0],.out=invalid[i][1]};char err[128]={0};size_t before=pack_allocation_calls;
        assert(!pack_linear(&l,err,sizeof(err)) && strstr(err,"invalid") && pack_allocation_calls==before && !l.packed && !l.packed_ld);
    }
    linear l={.in=32,.out=96};char err[128]={0};size_t before=pack_allocation_calls;
    fail_pack_allocation=true;assert(!pack_linear(&l,err,sizeof(err)));fail_pack_allocation=false;
    assert(pack_allocation_calls==before+1 && strstr(err,"out of memory") && !l.packed && !l.packed_ld);
    puts("head pack: 10 full-byte/special-value/guard shapes; 4 ragged, 5 invalid and allocation failure PASS");
}

int main(void) {
    check_pack_bytes();
    const int gelu_shapes[][2] = {{1024, 4096}, {4096, 1024}, {1024, 1024}, {1024, 896},
                                  {1025, 1280}, {8192, 192}, {63, 65}, {64, 64}};
    const int gelu_rows[] = {1, 2, 7};
    int gelu_checks = 0;
    for (size_t s = 0; s < sizeof(gelu_shapes) / sizeof(gelu_shapes[0]); s++)
        for (size_t r = 0; r < sizeof(gelu_rows) / sizeof(gelu_rows[0]); r++)
            for (int workers = 2; workers <= 4; workers += 2)
                for (int bias = 0; bias < 2; bias++) {
                    check_gelu4(gelu_shapes[s][0], gelu_shapes[s][1], gelu_rows[r], workers, bias);
                    gelu_checks += 2;
                }
    check_gelu4_bias_rounding();
    printf("head epilogue: %d exact cases; dispatch boundaries, guards, fallbacks and separate bias rounding OK\n", gelu_checks);
    const int scorer_rows[] = {1, 2, 3, 4, 9, 65};
    for (size_t i = 0; i < sizeof(scorer_rows) / sizeof(scorer_rows[0]); i++) check_scorer(scorer_rows[i]);
    const int shapes[][2] = {{63, 65}, {1024, 1024}, {4096, 1024}, {5120, 1024}, {1024, 4096}, {1024, 3072}};
    const int rows[] = {1, 2, 3, 4, 9, 32, 65, 127, 128, 129, 512, 513, 1024};
    int checks = 0;
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++)
        for (size_t r = 0; r < sizeof(rows) / sizeof(rows[0]); r++) {
            check(shapes[s][0], shapes[s][1], rows[r], 2);
            check(shapes[s][0], shapes[s][1], rows[r], 4); checks += 2;
        }
    /* Packing declines this otherwise four-worker-eligible width. The original
     * reduction must remain on two workers when the input width is ragged. */
    check(1025, 1280, 2, 4);
    check(1025, 1280, 7, 4);
    /* Packed output widths that allow two 32-aligned halves but not four. */
    check(8192, 192, 2, 4);
    check(8192, 192, 7, 4);
    checks += 4;
    char err[128];
    linear invalid = { .out = INT_MAX, .in = 1024 };
    assert(!pack_linear(&invalid, err, sizeof(err)) && strstr(err, "invalid"));
    assert(invalid.packed == NULL);
    linear failed = { .out = 64, .in = 64 };
    fail_pack_allocation = true;
    assert(!pack_linear(&failed, err, sizeof(err)) && strstr(err, "out of memory"));
    fail_pack_allocation = false;
    assert(failed.packed == NULL && failed.packed_ld == 0);

    /* Decoder self_in shares original QKV weights/bias, owns its packed copy. */
    clef_head *h = calloc(1, sizeof(*h)); assert(h);
    h->n_dec = 1;
    linear *q = &h->dec[0].self_attn.in_q;
    q->w = calloc(3 * 64 * 64, sizeof(float)); q->b = calloc(3 * 64, sizeof(float));
    assert(q->w && q->b);
    h->dec[0].self_in = (linear){ q->w, q->b, 3 * 64, 64, NULL, 0, 0 };
    assert(pack_linear(&h->dec[0].self_in, err, sizeof(err)));
    clef_head_free(h);
    printf("head linear: %d exact layout checks; single-row fallback, padding, scorer FP64 reference and ownership OK\n", checks);
    return 0;
}
