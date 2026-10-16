/* Model-free driver. Per-frame taps use the production frontend, not a clone. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float *captured[4][4];
static size_t captured_frames;
static void audio_stage(int clip, const char *name, size_t frame, const float *data, int n) {
    int stage = !strcmp(name,"real") ? 0 : !strcmp(name,"imag") ? 1 : !strcmp(name,"power") ? 2 : 3;
    if (captured[clip][stage]) for (int j = 0; j < n; j++) captured[clip][stage][j*captured_frames+frame] = data[j];
}
#define CLEF_AUDIO_STAGE audio_stage
#include "../clef_audio.c"

static bool write_stage(const char *folder, int clip, const char *name, const float *data, size_t n) {
    char path[4096];
    int length = snprintf(path,sizeof(path),"%s/%d-%s.bin",folder,clip,name);
    if (length < 0 || (size_t)length >= sizeof(path)) return false;
    FILE *f = fopen(path,"wb");
    if (!f) return false;
    bool ok = fwrite(data,4,n,f) == n;
    if (fclose(f)) ok = false;
    return ok;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 6) { fprintf(stderr,"usage: audio-features OUTPUT CLIP.f32 [CLIP.f32 ...]\n"); return 2; }
    const int n = argc-2;
    clef_audio_samples clips[4]={{0}};
    clef_audio_features features[4]={{0}};
    char err[256]="cannot read or write audio feature fixture";
    bool ok = false;
    for (int i = 0; i < n; i++) {
        FILE *f = fopen(argv[i+2],"rb");
        if (!f) goto done;
        if (fseek(f,0,SEEK_END)) { fclose(f); goto done; }
        long size = ftell(f);
        if (size < 0 || size > 19200000 || size%4 || fseek(f,0,SEEK_SET)) { fclose(f); goto done; }
        clips[i].count = (size_t)size/4;
        clips[i].data = malloc(size ? (size_t)size : 1);
        bool read_ok = clips[i].data && fread(clips[i].data,1,size,f) == (size_t)size && !ferror(f);
        if (fclose(f)) read_ok = false;
        if (!read_ok) goto done;
        if (clips[i].count/160 > captured_frames) captured_frames = clips[i].count/160;
    }
    for (int i = 0; i < n; i++) for (int s = 0; s < 4; s++) {
        captured[i][s] = calloc((s == 3 ? 128 : 201)*captured_frames+1,4);
        if (!captured[i][s]) goto done;
    }
    if (!clef_audio_mel_batch(clips,n,features,err,sizeof(err))) goto done;
    if (!write_stage(argv[1],0,"window",audio_hann,400)) goto done;
    {
        float filters[201*128]={0};
        for (int m = 0; m < 128; m++) for (int j = audio_mel_offsets[m]; j < audio_mel_offsets[m+1]; j++)
            filters[audio_mel_filters[j].bin*128+m] = audio_mel_filters[j].weight;
        if (!write_stage(argv[1],0,"filters",filters,201*128)) goto done;
    }
    for (int i = 0; i < n; i++) {
        const char *names[] = {"real","imag","power","mel"};
        for (int s = 0; s < 4; s++) if (!write_stage(argv[1],i,names[s],captured[i][s],(s == 3 ? 128 : 201)*captured_frames)) goto done;
        if (!write_stage(argv[1],i,"features",features[i].mel,128*captured_frames)) goto done;
    }
    printf("[");
    for (int i = 0; i < n; i++) printf("%s{\"frames\":%zu,\"padded_frames\":%zu,\"tokens\":%d}",
        i ? "," : "", features[i].frames,features[i].padded_frames,features[i].n_tokens);
    puts("]");
    ok = true;
done:
    for (int i = 0; i < n; i++) {
        clef_audio_samples_free(clips+i); clef_audio_features_free(features+i);
        for (int s = 0; s < 4; s++) free(captured[i][s]);
    }
    if (!ok) fprintf(stderr,"%s\n",err);
    return ok ? 0 : 1;
}
