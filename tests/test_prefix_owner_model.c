/* Real-model cache ownership and keep-warm regression. Shared GPU lock required. */
#include "../clef_engine.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern bool clef_run_ex(clef_engine *, const clef_record *, int, float ****, bool, float *, int, char *, size_t);

static void need(bool ok,const char *message) {
    if(!ok) { fprintf(stderr,"ownership model check: %s\n",message); exit(1); }
}
static void same(const clef_record *rec,float ***a,float ***b) {
    for(int q=0;q<rec->nq;q++) {
        for(int k=0;k<rec->q[q].n_opt;k++) need(isfinite(a[0][q][k])&&isfinite(b[0][q][k]),"nonfinite logits");
        need(!memcmp(a[0][q],b[0][q],(size_t)rec->q[q].n_opt*sizeof(float)),"cache or keep-warm changed logit bits");
    }
}
static void wrong_owner(clef_engine *e,clef_prefix *p,const clef_record *rec) {
    char err[256]={0};
    float ***out=NULL;
    int reused=99;
    need(!clef_run_prefix(e,p,rec,&out,true,&reused,err,sizeof(err)),"wrong owner was accepted");
    need(strstr(err,"belongs to another engine")!=NULL,"missing ownership error");
    need(out==NULL&&reused==0,"failed owner check changed output/reuse");
    need(!clef_prefix_keep_warm(e,p,err,sizeof(err)),"keep-warm accepted wrong owner");
    need(strstr(err,"belongs to another engine")!=NULL,"missing keep-warm ownership error");
}
int main(int argc,char **argv) {
    need(argc==4,"usage: ownership-model-check MODEL_A MODEL_B REQUEST.jsonl");
    char err[512]={0};
    clef_engine *first=clef_open(argv[1],err,sizeof(err));
    need(first!=NULL,err);
    const uint64_t identity=first->instance_id;
    const uintptr_t address=(uintptr_t)first;
    need(identity!=0,"engine has no identity");
    FILE *f=fopen(argv[3],"r");
    need(f!=NULL,"cannot open request");
    char *line=NULL;
    size_t cap=0;
    ssize_t length=getline(&line,&cap,f);
    need(length>0,"empty request");
    need(fgetc(f)==EOF&&!ferror(f)&&fclose(f)==0,"expected one complete request");
    jarena *arena=jarena_new();
    need(arena!=NULL,"cannot allocate arena");
    jval *request=json_parse(arena,line,(size_t)length,err,sizeof(err));
    need(request!=NULL,err);
    clef_encode_opts opts=CLEF_ENCODE_DEFAULTS;
    opts.strict=true; opts.reject_truncation=true;
    clef_record rec={0};
    need(clef_encode_request(first->tok,request,opts,&rec,err,sizeof(err)),err);
    float ***reference=NULL;
    need(clef_run_ex(first,&rec,1,&reference,true,NULL,0,err,sizeof(err)),err);
    clef_prefix *entry=clef_prefix_new();
    need(entry!=NULL,"cannot allocate entry");
    for(int pass=0;pass<2;pass++) {
        float ***got=NULL;
        int reused=-1;
        need(clef_run_prefix(first,entry,&rec,&got,true,&reused,err,sizeof(err)),err);
        need(pass?reused>0:reused==0,"incorrect initial fill/hit");
        same(&rec,reference,got);
        clef_free_probs(&rec,1,got);
    }
    need(clef_prefix_bytes(entry)>0,"test did not populate a GPU cache");
    clef_engine *other=clef_open(argv[2],err,sizeof(err));
    need(other!=NULL,err);
    need(other->instance_id!=identity,"two model opens share an identity");
    wrong_owner(other,entry,&rec);
    need(clef_prefix_keep_warm(first,entry,err,sizeof(err)),err);
    float ***got=NULL;
    int reused=0;
    need(clef_run_prefix(first,entry,&rec,&got,true,&reused,err,sizeof(err)),err);
    need(reused>0,"wrong-owner failure discarded the original cache");
    same(&rec,reference,got);
    clef_free_probs(&rec,1,got);
    clef_close(other);
    clef_close(first);
    clef_engine *reopened=clef_open(argv[1],err,sizeof(err));
    need(reopened!=NULL,err);
    need(reopened->instance_id!=identity,"reopened engine reused identity");
    wrong_owner(reopened,entry,&rec);
    printf("{\"pass\":true,\"tokens\":%zu,\"cache_bytes\":%zu,\"reused_tokens\":%d,"
           "\"cross_model_rejected\":true,\"reopened_model_rejected\":true,\"original_cache_preserved\":true,"
           "\"keep_warm_logits_identical\":true,\"address_recycled\":%s}\n",rec.ids.len,clef_prefix_bytes(entry),
           reused,(uintptr_t)reopened==address?"true":"false");
    clef_prefix_free(entry);
    clef_free_probs(&rec,1,reference);
    clef_record_free(&rec);
    jarena_free(arena);
    free(line);
    clef_close(reopened);
    return 0;
}
