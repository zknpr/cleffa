/* Owned audio preprocessing must unwind partial records at every allocation. */
#include <assert.h>
#include <stdlib.h>
#include <string.h>
static int calls,fail_at;
static void *record_malloc(size_t n){return ++calls==fail_at?NULL:malloc(n);}
static void *record_calloc(size_t n,size_t s){return ++calls==fail_at?NULL:calloc(n,s);}
#define malloc record_malloc
#define calloc record_calloc
#include "../clef_record_omni.c"
#undef malloc
#undef calloc

int main(void) {
    jbuf text={0};jbuf_puts(&text,"[");
    for(int clip=0;clip<2;clip++) {
        jbuf_puts(&text,clip?",[":"[");
        for(int sample=0;sample<201;sample++)jbuf_puts(&text,sample?",0":"0");
        jbuf_puts(&text,"]");
    }
    jbuf_puts(&text,"]");
    jarena *arena=jarena_new();char err[256];
    jval *audio=json_parse(arena,text.p,text.len,err,sizeof(err));assert(audio);
    clef_encode_opts opts=CLEF_ENCODE_DEFAULTS;opts.max_length=150;opts.audio.token_id=1;
    clef_record record={0};size_t reserve=100;
    assert(clef_omni_encode_audio(audio,0,&opts,&reserve,&record,err,sizeof(err)));
    assert(record.n_audio==2 && record.n_audio_tokens==2 && reserve==108);
    const int allocations=calls;assert(allocations==5);
    clef_record_free(&record);
    for(int fail=1;fail<=allocations;fail++) {
        calls=0;fail_at=fail;reserve=100;
        assert(!clef_omni_encode_audio(audio,0,&opts,&reserve,&record,err,sizeof(err)) && reserve==100);
        clef_record_free(&record);assert(!record.audio && !record.n_audio);
    }
    fail_at=0;reserve=100;
    opts.audio.max_samples=401;
    assert(!clef_omni_encode_audio(audio,0,&opts,&reserve,&record,err,sizeof(err)));
    clef_record_free(&record);
    opts.audio.max_samples=402;
    assert(clef_omni_encode_audio(audio,0,&opts,&reserve,&record,err,sizeof(err)));
    clef_record_free(&record);
    calls=0;reserve=100;opts.max_length=107;
    assert(!clef_omni_encode_audio(audio,0,&opts,&reserve,&record,err,sizeof(err)) && calls==0);
    opts.max_length=150;opts.audio.max_items=1;
    assert(!clef_omni_encode_audio(audio,0,&opts,&reserve,&record,err,sizeof(err)) && calls==0);
    /* Joined soundtrack entries borrow JSON values. Partial allocation must
     * leave the request arena reusable and free every owned feature/row map. */
    jmember video_members[]={{"audio",5,audio->items[1]}};
    jval video={.type=J_OBJECT,.members=video_members,.n=1};jval *video_items[]={&video};
    jval videos={.type=J_ARRAY,.items=video_items,.n=1};
    jval standalone={.type=J_ARRAY,.items=audio->items,.n=1};
    jmember members[]={{"audio",5,&standalone},{"videos",6,&videos}};
    jval request={.type=J_OBJECT,.members=members,.n=2};
    opts.audio.max_items=2;calls=0;reserve=100;
    assert(clef_omni_encode_record_audio(&request,&opts,&reserve,&record,err,sizeof(err)));
    assert(calls==6 && reserve==107 && record.audio[0].video_index==-1 && record.audio[1].video_index==0);
    clef_record_free(&record);
    for(int fail=1;fail<=6;fail++) {
        calls=0;fail_at=fail;reserve=100;
        assert(!clef_omni_encode_record_audio(&request,&opts,&reserve,&record,err,sizeof(err)) && reserve==100);
        clef_record_free(&record);
    }
    fail_at=0;reserve=100;
    assert(clef_omni_encode_record_audio(&request,&opts,&reserve,&record,err,sizeof(err)));
    clef_record_free(&record);
    jarena_free(arena);jbuf_free(&text);
    puts("audio record: five standalone/six soundtrack allocation failures, recovery and pre-allocation limits passed");
    return 0;
}
