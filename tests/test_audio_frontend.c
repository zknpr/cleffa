#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include "../clef_audio.h"
static int allocation, fail_at;
static void *test_malloc(size_t n) { return ++allocation==fail_at ? NULL : malloc(n); }
static void *test_calloc(size_t n,size_t s) { return ++allocation==fail_at ? NULL : calloc(n,s); }
#define malloc test_malloc
#define calloc test_calloc
#include "../clef_audio.c"
#undef malloc
#undef calloc

int main(void) {
    float short_clip[1]={.5f}, wave[401]={0};
    clef_audio_samples clips[]={{short_clip,1},{wave,401}};
    clef_audio_features out[2]={{0}};
    char err[256];
    assert(!clef_audio_mel_batch(clips,1,out,err,sizeof(err)) && !out[0].mel);
    for (int at=1;at<=3;at++) {
        fail_at=at;allocation=0;
        assert(!clef_audio_mel_batch(clips,2,out,err,sizeof(err)) && !out[0].mel && !out[1].mel);
    }
    fail_at=0;
    assert(clef_audio_mel_batch(clips,2,out,err,sizeof(err)));
    assert(out[0].frames==1 && out[1].frames==2 && out[0].padded_frames==2 && out[1].padded_frames==2);
    assert(out[0].n_tokens==1 && out[1].n_tokens==1);
    for(int i=0;i<256;i++){assert(isfinite(out[0].mel[i]));assert(out[1].mel[i]==-1.5f);}
    for(int i=0;i<2;i++)clef_audio_features_free(out+i);
    wave[3]=INFINITY;
    assert(!clef_audio_mel_batch(clips,2,out,err,sizeof(err)) && !out[0].mel && !out[1].mel);
    wave[3]=FLT_MAX;
    assert(!clef_audio_mel_batch(clips,2,out,err,sizeof(err)) && !out[0].mel && !out[1].mel);
    clef_audio_features features[]={{wave,801,801,105},{wave,99,801,13}};
    clef_audio_layout layout={0};
    for(int at=1;at<=2;at++) {
        fail_at=at;allocation=0;
        assert(!clef_audio_layout_make(features,2,&layout,err,sizeof(err)));
        assert(!layout.chunks && !layout.token_position && !layout.tokens);
    }
    fail_at=0;
    assert(clef_audio_layout_make(features,2,&layout,err,sizeof(err)));
    assert(layout.n_chunks==10 && layout.tokens==118 && layout.max_chunk_frames==100);
    assert(layout.window_end[103]==104 && layout.window_start[104]==104 && layout.window_end[104]==105);
    assert(layout.window_start[105]==105 && layout.window_end[117]==118);
    clef_audio_layout_free(&layout);
    assert(!layout.chunks && !layout.token_position && !layout.tokens);
    features[0].n_tokens++;
    assert(!clef_audio_layout_make(features,2,&layout,err,sizeof(err)) && !layout.chunks);
    features[0].n_tokens--; features[1].padded_frames=100;
    assert(!clef_audio_layout_make(features,2,&layout,err,sizeof(err)) && !layout.chunks);
    features[1].padded_frames=801; features[0].frames=SIZE_MAX;
    assert(!clef_audio_layout_make(features,2,&layout,err,sizeof(err)) && !layout.chunks);
    return 0;
}
