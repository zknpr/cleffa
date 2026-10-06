/* CPU-only ownership regression: no model or Metal device is opened.
 * Including the implementation gives the test the same engine-ID allocator as clef_open. */
#include "../clef.c"
#include <assert.h>

static void rejects(clef_engine *e, clef_prefix *p, const char *message) {
    clef_record empty={0};
    float ***out=(float ***)(uintptr_t)0x1234;
    int reused=99;
    char err[256]={0};
    assert(!clef_run_prefix(e,p,&empty,&out,true,&reused,err,sizeof(err)));
    assert(strstr(err,message));
    assert(reused==0);
    assert(out==(float ***)(uintptr_t)0x1234);
    reused=99;
    assert(!clef_run_template(e,p,&empty,&out,true,&reused,err,sizeof(err)));
    assert(strstr(err,message));
    assert(reused==0);
    assert(out==(float ***)(uintptr_t)0x1234);
}

static void rejects_template_boundaries(clef_engine *e, clef_prefix *p, const char *message) {
    const size_t lengths[]={768, 1025, 1088, 1089, 2049};
    for (size_t i=0; i<sizeof(lengths)/sizeof(lengths[0]); i++) {
        clef_record rec={0};
        rec.ids.len=lengths[i];
        rec.schema_start=40;
        float ***out=(float ***)(uintptr_t)0x1234;
        int reused=99;
        char err[256]={0};
        /* Both eligible and bypassed requests reject the owner before touching token/GPU
         * buffers. These mocks deliberately have neither, so accidental use also fails. */
        assert(!clef_run_template(e,p,&rec,&out,true,&reused,err,sizeof(err)));
        assert(strstr(err,message));
        assert(reused==0);
        assert(out==(float ***)(uintptr_t)0x1234);
    }
}

int main(void) {
    clef_engine first={0}, second={0};
    first.cfg.H=5120;
    second.cfg.H=4096;
    assert(claim_engine_id(&first.instance_id));
    assert(claim_engine_id(&second.instance_id));
    assert(first.instance_id && second.instance_id && first.instance_id!=second.instance_id);
    const uint64_t original=first.instance_id;
    clef_prefix *p=clef_prefix_new();
    assert(p);
    char err[256]={0};
    assert(clef_prefix_keep_warm(&first,p,err,sizeof(err)));  /* empty entry; no GPU call */
    rejects(&first,p,"bad length");  /* binds before the ordinary length validation fails */
    assert(p->engine_id==original);
    rejects(&second,p,"belongs to another engine");
    rejects_template_boundaries(&second,p,"belongs to another engine");
    rejects_template_boundaries(NULL,p,"invalid prefix cache owner");
    rejects_template_boundaries(&first,NULL,"invalid prefix cache owner");
    assert(!clef_prefix_keep_warm(&second,p,err,sizeof(err)));
    assert(strstr(err,"belongs to another engine"));
    assert(p->engine_id==original);
    assert(claim_engine_id(&first.instance_id));  /* simulate an engine reopened at the same address */
    rejects(&first,p,"belongs to another engine");
    rejects_template_boundaries(&first,p,"belongs to another engine");
    assert(!clef_prefix_keep_warm(&first,p,err,sizeof(err)));
    first.instance_id=original;
    rejects(&first,p,"bad length");
    assert(clef_prefix_keep_warm(&first,p,err,sizeof(err)));
    assert(clef_prefix_bytes(p)==0);
    /* NULL handles are rejected explicitly, as clef_prefix_free accepts them. The volatile
       pointers keep the compiler from folding a constant NULL dereference into unreachable code. */
    clef_prefix *volatile no_prefix=NULL;
    clef_engine *volatile no_engine=NULL;
    assert(clef_prefix_bytes(no_prefix)==0);
    err[0]=0;
    assert(!clef_keep_warm(no_engine,err,sizeof(err)) && strstr(err,"invalid engine"));
    clef_prefix_free(p);
    puts("PASS: cache binds before buffer use; other engines and recycled addresses fail explicitly; empty keep-warm performs no GPU work");
    return 0;
}
