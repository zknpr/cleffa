/* Wrap only the CLI's allocations, leaving model/Metal allocation untouched.
 * Exercise each ownership transition with two ordinary requests. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../clef_engine.h"

bool clef_run_ex(clef_engine *e, const clef_record *recs, int n, float ****out, bool raw,
                 float *dump, int rows, char *err, size_t errlen);

static int alloc_count, fail_at, close_count;
static int arena_count, fail_arena_at, live_arenas, run_count, fail_run_at;
static void *live[4];

static void *test_calloc(size_t n, size_t size) {
    int call = ++alloc_count;
    if (call == fail_at) return NULL;
    void *p = calloc(n, size);
    if (call <= 4) live[call - 1] = p; /* batch arrays and optional dump */
    return p;
}

static void test_free(void *p) {
    for (int i = 0; i < 4; i++) if (live[i] == p) live[i] = NULL;
    free(p);
}

static void test_close(clef_engine *e) {
    close_count++;
    clef_close(e);
}

static jarena *test_arena_new(void) {
    if (++arena_count == fail_arena_at) return NULL;
    jarena *a = jarena_new();
    if (a) live_arenas++;
    return a;
}

static void test_arena_free(jarena *a) {
    if (a) live_arenas--;
    jarena_free(a);
}

static bool test_run(clef_engine *e, const clef_record *recs, int n, float ****out,
                     bool raw, float *dump, int rows, char *err, size_t errlen) {
    if (++run_count == fail_run_at) {
        snprintf(err, errlen, "injected inference failure");
        return false;
    }
    return clef_run_ex(e, recs, n, out, raw, dump, rows, err, errlen);
}

#define main clef_cli_main
#define calloc test_calloc
#define free test_free
#define clef_close test_close
#define jarena_new test_arena_new
#define jarena_free test_arena_free
#define clef_run_ex test_run
#include "../clef_main.c"
#undef clef_run_ex
#undef jarena_free
#undef jarena_new
#undef clef_close
#undef free
#undef calloc
#undef main

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: test-cli-alloc MODEL.gguf\n"); return 2; }
    char input[] = "/tmp/clef-alloc-input.XXXXXX", dump[] = "/tmp/clef-alloc-dump.XXXXXX";
    int fd = mkstemp(input), df = mkstemp(dump);
    assert(fd >= 0 && df >= 0);
    close(df);
    FILE *request = fdopen(fd, "w");
    const char *text = "{\"model\":\"m\",\"state\":\"test\",\"questions\":{\"q\":{\"type\":\"noul\"}}}\n";
    assert(fputs(text, request) >= 0 && fputs(text, request) >= 0);
    assert(fclose(request) == 0);
    FILE *output = tmpfile(), *errors = tmpfile();
    assert(output && errors);
    int saved_out = dup(STDOUT_FILENO), saved_err = dup(STDERR_FILENO);
    assert(saved_out >= 0 && saved_err >= 0);
    char *args[] = { "clef", "-m", argv[1], "--batch", "2", "--dump", dump, input, NULL };
    for (int failure = 1; failure <= 9; failure++) {
        assert(dup2(fileno(output), STDOUT_FILENO) >= 0 && dup2(fileno(errors), STDERR_FILENO) >= 0);
        rewind(output); rewind(errors);
        assert(ftruncate(fileno(output), 0) == 0 && ftruncate(fileno(errors), 0) == 0);
        alloc_count = close_count = 0;
        arena_count = live_arenas = run_count = 0;
        fail_at = failure <= 5 ? failure : 0;
        fail_arena_at = failure == 6 ? 1 : failure == 7 ? 2 : 0;
        fail_run_at = failure == 8 ? 2 : 0;
        bool injected = failure < 9;
        memset(live, 0, sizeof(live));
        int rc = clef_cli_main(8, args);
        fflush(stdout); fflush(stderr);
        assert(dup2(saved_out, STDOUT_FILENO) >= 0 && dup2(saved_err, STDERR_FILENO) >= 0);
        assert(rc == (injected ? 1 : 0));
        assert(close_count == 1 && live_arenas == 0);
        for (int i = 0; i < 4; i++) assert(!live[i]);
        rewind(errors);
        char message[512] = {0};
        assert(fread(message, 1, sizeof(message) - 1, errors) < sizeof(message));
        if (fail_run_at) assert(strstr(message, "injected inference failure"));
        else if (injected) assert(strstr(message, "memory") || strstr(message, "allocate dump"));
        else {
            rewind(output);
            char line[1024];
            for (int i = 0; i < 2; i++) assert(fgets(line, sizeof(line), output) && strstr(line, "\"answers\""));
            assert(!fgets(line, sizeof(line), output));
        }
        printf("CLI allocation %d: %s and cleanup passed\n", failure, injected ? "error propagation" : "normal dump/batch");
        fflush(stdout);
    }
    close(saved_out); close(saved_err);
    fclose(output); fclose(errors);
    unlink(input); unlink(dump);
    return 0;
}
