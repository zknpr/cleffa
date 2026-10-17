/* Model-free position bounds, temporal origins and allocation failure. */
#include <assert.h>
#include <stdlib.h>
#include <string.h>
static int fail_alloc;
static void *position_malloc(size_t n) { return fail_alloc ? NULL : malloc(n); }
#define malloc position_malloc
#include "../clef_record_omni.c"
#undef malloc

int main(void) {
    clef_image_ref images[3] = {
        {.tok_start=2,.pt={.grid_h=4,.grid_w=6,.n_tokens=6}},
        {.tok_start=11,.video_group=1,.pt={.grid_h=4,.grid_w=6,.n_tokens=6}},
        {.tok_start=17,.video_group=1,.pt={.grid_h=4,.grid_w=6,.n_tokens=6}},
    };
    clef_record record={.ids={.len=27},.images=images,.n_images=3};
    char err[128];
    fail_alloc=1;
    assert(!clef_omni_positions(&record,err,sizeof(err)) && !record.pos3);
    assert(strstr(err,"out of memory"));
    fail_alloc=0;
    assert(clef_omni_positions(&record,err,sizeof(err)));
    const int32_t first_video=record.pos3[3*11];
    assert(record.pos3[3*17]==first_video+13);
    assert(record.pos3[3*17+1]==record.pos3[3*11+1]);
    assert(record.pos3[3*17+2]==record.pos3[3*11+2]);
    assert(record.pos3[3*23]==first_video+14);
    for(int t=0;t<27;t++) for(int a=0;a<3;a++) assert(record.pos3[3*t+a]>=0);
    assert(!clef_omni_positions(&record,err,sizeof(err))); /* no replacement leak */
    free(record.pos3);record.pos3=NULL;
    for(int which=0;which<4;which++) {
        clef_image_ref saved=images[2];
        if(which==0) images[2].tok_start=16;
        if(which==1) images[2].pt.n_tokens=7;
        if(which==2) images[2].pt.grid_h=INT_MAX;
        if(which==3) record.ids.len=18;
        assert(!clef_omni_positions(&record,err,sizeof(err)) && !record.pos3);
        images[2]=saved;record.ids.len=27;
    }
    /* A trailing audio segment controls EOS even if prior spatial IDs exceed
     * its final time. Both audio wrappers occupy ordinary text positions. */
    int32_t audio_rows[]={34};
    clef_audio_ref track={.feat={.n_tokens=1},.token_rows=audio_rows,.video_index=0};
    clef_image_ref wide={.tok_start=2,.video_group=1,.pt={.grid_h=2,.grid_w=64,.n_tokens=32}};
    record=(clef_record){.ids={.len=38},.images=&wide,.n_images=1,.audio=&track,.n_audio=1};
    assert(clef_omni_positions(&record,err,sizeof(err)));
    assert(record.pos3[3*33+2]==33 && record.pos3[3*34]==2 && record.pos3[3*35]==3);
    free(record.pos3);record.pos3=NULL;audio_rows[0]--;
    assert(!clef_omni_positions(&record,err,sizeof(err)) && !record.pos3);
    puts("Omni positions: temporal origins, placement bounds and allocation recovery passed");
    return 0;
}
