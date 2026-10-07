/* CPU-only checkpoint planner regression. The mocked GPU stores exact token identities
 * and a deterministic recurrent checksum; it never creates a Metal device or opens a model.
 * This verifies state selection and failure recovery, not real GPU numerical equivalence. */
#define clef_gpu_prefix_new mock_prefix_new
#define clef_gpu_prefix_free mock_prefix_free
#define clef_gpu_prefix_bytes mock_prefix_bytes
#define clef_gpu_prefix_supported mock_prefix_supported
#define clef_gpu_prefix_class mock_prefix_class
#define clef_gpu_forward_prefix mock_forward_prefix
#define clef_gpu_forward mock_forward
#define clef_head_run mock_head_run
#include "../clef.c"
#include <assert.h>

struct clef_gpu_prefix {
    struct { int row, cls; int32_t *ids; uint64_t sum; } slot[CLEF_PREFIX_CKPT];
};
static int failure;
static float result;
static unsigned loads, stores, passes, failures[6], bf16_reruns, multi_store, largest_plan;
static uint64_t advance(uint64_t s, int32_t id) { return (s ^ (uint32_t)id) * UINT64_C(1099511628211); }
static uint64_t checksum(const int32_t *ids,int n) {
    uint64_t h=UINT64_C(1469598103934665603);
    for(int i=0;i<n;i++)h=advance(h,ids[i]);
    return h;
}
static float score(uint64_t sum) { return (float)(sum & 0xffffffu); }
clef_gpu_prefix *mock_prefix_new(void) { return calloc(1,sizeof(clef_gpu_prefix)); }
void mock_prefix_free(clef_gpu_prefix *p) {
    if(!p)return;
    for(int i=0;i<CLEF_PREFIX_CKPT;i++)free(p->slot[i].ids);
    free(p);
}
size_t mock_prefix_bytes(const clef_gpu_prefix *p) {
    size_t n=sizeof(*p);
    for(int i=0;i<CLEF_PREFIX_CKPT;i++)n+=(size_t)p->slot[i].row*4;
    return n;
}
bool mock_prefix_supported(const clef_gpu *g) { (void)g;return true; }
int mock_prefix_class(const clef_engine *e,int length) { return e->cfg.H==5120 && length>=4096; }

bool mock_forward_prefix(clef_gpu *g,const clef_engine *e,clef_gpu_prefix *p,const int32_t *ids,int T,
                         int L,const clef_prefix_plan *plan,bool *overflow,clef_head_inputs *in,char *err,size_t errlen) {
    (void)g;passes++;
    assert(L>=0 && L<T && L%32==0);
    assert(plan->n>=0 && plan->n<CLEF_PREFIX_CKPT);
    assert(L>0 || plan->n>0);
    if((unsigned)plan->n>largest_plan)largest_plan=(unsigned)plan->n;
    if(plan->n>1)multi_store++;
    uint64_t state=UINT64_C(1469598103934665603);
    if(L) {
        assert(plan->load>=0 && plan->load<CLEF_PREFIX_CKPT);
        assert(p->slot[plan->load].row==L);
        assert(p->slot[plan->load].cls==mock_prefix_class(e,T));
        /* Independent of eviction policy: every reused token must be this request's token. */
        if(getenv("CLEF_TEST_CKPT_CORRUPT") && loads==0)p->slot[plan->load].ids[0]^=1;
        if(memcmp(p->slot[plan->load].ids,ids,(size_t)L*4)) {
            fputs("REJECT: reused checkpoint belongs to different tokens\n",stderr);
            exit(23);
        }
        assert(p->slot[plan->load].sum==checksum(ids,L));
        state=p->slot[plan->load].sum;loads++;
    }
    if(failure==1) { failures[1]++;snprintf(err,errlen,"mock allocation failure");return false; }
    int row=L;
    for(int i=0;i<plan->n;i++) {
        int s=plan->slot[i],at=plan->row[i];
        assert(s>=0 && s<CLEF_PREFIX_CKPT && (!L || s!=plan->load));
        for(int j=0;j<i;j++)assert(s!=plan->slot[j]);
        assert(at>row && at<T && at%32==0);
        for(;row<at;row++)state=advance(state,ids[row]);
        int32_t *copy=realloc(p->slot[s].ids,(size_t)at*4);assert(copy);
        p->slot[s].ids=copy;memcpy(copy,ids,(size_t)at*4);
        p->slot[s].row=at;p->slot[s].cls=mock_prefix_class(e,T);p->slot[s].sum=state;stores++;
        if(failure==2) { failures[2]++;snprintf(err,errlen,"mock failure after one state write");return false; }
    }
    if(failure==2 || failure==3) { failures[failure]++;snprintf(err,errlen,"mock execution failure");return false; }
    for(;row<T;row++)state=advance(state,ids[row]);
    assert(state==checksum(ids,T));
    result=score(state);in->nh=&result;
    *overflow=failure==4;
    if(*overflow)failures[4]++;
    return true;
}

bool mock_forward(clef_gpu *g,const clef_engine *e,const int32_t *ids,const int32_t *pos,
                  const int32_t *ss,const int32_t *bounds,int n_seq,int T,bool bf16_only,bool *overflow,
                  clef_head_inputs *in,float *dump,int dump_rows,char *err,size_t errlen) {
    (void)g;(void)e;(void)pos;(void)ss;(void)bounds;(void)dump;(void)dump_rows;(void)err;(void)errlen;
    assert(n_seq==1);
    if(bf16_only)bf16_reruns++;
    if(overflow)*overflow=failure==4 && !bf16_only;
    result=score(checksum(ids,T));in->nh=&result;
    return true;
}
bool mock_head_run(const clef_head *head,const clef_config *cfg,const gguf_tensor *lm,
                   const clef_head_inputs *in,int row,const clef_record *rec,float **out) {
    (void)head;(void)cfg;(void)lm;(void)row;
    assert(rec->nq==1 && rec->q[0].n_opt==1);
    if(failure==5) { failures[5]++;return false; }
    out[0][0]=in->nh[0];return true;
}
static uint32_t random_state=0x75031;
static uint32_t next_random(void) { random_state^=random_state<<13;random_state^=random_state>>17;random_state^=random_state<<5;return random_state; }

int main(void) {
    const int capacity=1<<20;
    int32_t *ids=malloc((size_t)capacity*4);assert(ids);
    for(int model=0;model<2;model++) {
        clef_engine e={0};e.cfg.H=model?5120:4096;e.cfg.vocab=65536;assert(claim_engine_id(&e.instance_id));
        clef_prefix *p=clef_prefix_new();assert(p);
        int T=16347;
        for(int i=0;i<capacity;i++)ids[i]=(int32_t)(next_random()&65535);
        for(int iteration=0;iteration<3000;iteration++) {
            int kind=iteration%10;
            if(kind==0) {
                static const int sizes[]={346,2047,2048,2049,4095,4096,4097,8072,16347};
                T=sizes[next_random()%9];
                for(int i=36;i<T;i++)ids[i]=(int32_t)(next_random()&65535);
            } else if(kind==2 || kind==3 || kind==9) {
                int at=36+(int)(next_random()%(unsigned)(T-36));ids[at]^=1;
            } else if(kind==4) {
                T=256+(int)(next_random()%(unsigned)(T-255));
            } else if(kind==5) {
                T=256+(int)(next_random()%16092u);
            }
            int schema=T-96-(kind==6?64:0);
            int32_t opt[1][2]={{T-50,T-40}};
            clef_question question={.span={T-90,T-70},.n_opt=1,.opt_span=opt};
            clef_record rec={.ids={.ids=ids,.len=(size_t)T},.schema_start=schema,.q=&question,.nq=1};
            failure=kind==7?1+(iteration/10)%5:0;
            char err[256]={0};float ***out=NULL;int reused=99;
            bool ok=clef_run_prefix(&e,p,&rec,&out,true,&reused,err,sizeof(err));
            if(failure && failure!=4) {
                assert(!ok && err[0] && !out && reused==0 && p->len==0);
            } else {
                assert(ok && !err[0] && out && out[0][0][0]==score(checksum(ids,T)));
                assert(reused>=0 && reused<T && reused%32==0);
                if(failure==4)assert(reused==0 && p->len==0);
                clef_free_probs(&rec,1,out);
            }
        }
        /* Exercise period growth at the library's accepted upper length, beyond the HTTP
         * model fixture sizes. Only a bounded number of checkpoint slots may be planned. */
        failure=0;
        clef_prefix_free(p);p=clef_prefix_new();assert(p);
        for(int step=0;step<3;step++) {
            const int size=20584;
            if(step)ids[step==1?640:768]^=1;
            int32_t opt[1][2]={{size-50,size-40}};
            clef_question q={.span={size-90,size-70},.n_opt=1,.opt_span=opt};
            clef_record rec={.ids={.ids=ids,.len=(size_t)size},.schema_start=size-96,.q=&q,.nq=1};
            float ***out=NULL;int reused=0;char err[256];
            assert(clef_run_prefix(&e,p,&rec,&out,true,&reused,err,sizeof(err)));
            assert(out[0][0][0]==score(checksum(ids,size)));
            if(step==2)assert(reused==640); /* one loaded slot beside eleven stored rows */
            clef_free_probs(&rec,1,out);
        }
        for(int size=32768;size<=capacity;size*=2) {
            int32_t opt[1][2]={{size-50,size-40}};
            clef_question q={.span={size-90,size-70},.n_opt=1,.opt_span=opt};
            clef_record rec={.ids={.ids=ids,.len=(size_t)size},.schema_start=size-96,.q=&q,.nq=1};
            float ***out=NULL;int reused=0;char err[256];
            assert(clef_run_prefix(&e,p,&rec,&out,true,&reused,err,sizeof(err)));
            assert(out[0][0][0]==score(checksum(ids,size)));
            clef_free_probs(&rec,1,out);
        }
        clef_prefix_free(p);
    }
    free(ids);
    assert(loads>100 && stores>100 && multi_store>100 && largest_plan==11);
    for(int i=1;i<=5;i++)assert(failures[i]>0);
    assert(bf16_reruns==failures[4]);
    printf("PASS CPU planner: %u prefix passes, %u safe resumes, %u state stores, %u multistore plans, max %u slots; failures %u/%u/%u/%u/%u, BF16 reruns %u\n",passes,loads,stores,multi_store,largest_plan,failures[1],failures[2],failures[3],failures[4],failures[5],bf16_reruns);
    return 0;
}
