#include <assert.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <unistd.h>
#include "../clef_dump.h"

int main(void) {
    float input[] = {1.25f, -3.5f, 0.0f, 8.0f}, memory[8] = {0}, readback[8];
    char err[256];
    FILE *file = tmpfile();
    assert(file);
    clef_dump disk = {.file = file, .elements = 8};
    clef_dump ram = {.memory = memory, .elements = 8};
    /* A retry overwrites the same layer instead of appending a second pass. */
    for (int pass = 0; pass < 2; pass++) {
        input[0] += 1;
        for (size_t layer = 0; layer < 2; layer++) {
            assert(clef_dump_write(&disk, layer * 4, input, 4, err, sizeof(err)));
            assert(clef_dump_write(&ram, layer * 4, input, 4, err, sizeof(err)));
        }
    }
    rewind(file);
    assert(fread(readback, sizeof(float), 8, file) == 8 && fgetc(file) == EOF);
    assert(!memcmp(readback, memory, sizeof(memory)));
    assert(!clef_dump_write(&disk, 7, input, 4, err, sizeof(err)));
    assert(!clef_dump_write(&disk, SIZE_MAX, input, 4, err, sizeof(err)));
    assert(fclose(file) == 0);
    /* Buffered writes must fail here, before a later layer or output publishes. */
    file = tmpfile(); assert(file); disk.file = file;
    struct rlimit limit = {0, 0};
    assert(signal(SIGXFSZ, SIG_IGN) != SIG_ERR);
    assert(setrlimit(RLIMIT_FSIZE, &limit) == 0);
    assert(!clef_dump_write(&disk, 0, input, 4, err, sizeof(err)));
    assert(strstr(err, "incomplete dump"));
    fclose(file);
    puts("residual dump: byte parity, retry overwrite, bounds and write failure passed");
    return 0;
}
