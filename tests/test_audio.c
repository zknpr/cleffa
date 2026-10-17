/* WAV/JSON boundaries and allocation order under ASan/UBSan. */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../clef_audio.h"
static int allocations,fail_alloc;
static void *audio_alloc(size_t n) { allocations++;return fail_alloc?NULL:malloc(n); }
#define malloc audio_alloc
#include "../clef_audio.c"
#undef malloc

static void u16(uint8_t *p,unsigned v){p[0]=v;p[1]=v>>8;}
static void u32(uint8_t *p,uint32_t v){for(int i=0;i<4;i++)p[i]=v>>(8*i);}
static size_t wave(uint8_t *p,int type,int count) {
    int width=type==1?2:4;memset(p,0,256);memcpy(p,"RIFF",4);u32(p+4,36+count*width);
    memcpy(p+8,"WAVEfmt ",8);u32(p+16,16);u16(p+20,type);u16(p+22,1);u32(p+24,16000);
    u32(p+28,16000*width);u16(p+32,width);u16(p+34,width*8);memcpy(p+36,"data",4);u32(p+40,count*width);
    return 44+count*width;
}
static void refuse(const uint8_t *p,size_t n,size_t limit) {
    clef_audio_samples out={0};char err[128]={0};int before=allocations;
    assert(!clef_audio_decode(p,n,limit,&out,err,sizeof(err)) && !out.data && !out.count && err[0]);
    assert(allocations==before);
}
int main(void) {
    uint8_t p[256],q[256];char err[128];clef_audio_samples out={0};
    size_t n=wave(p,1,3);u16(p+44,32768);u16(p+46,0);u16(p+48,32767);
    assert(clef_audio_decode(p,n,3,&out,err,sizeof(err)) && out.count==3);
    assert(out.data[0]==-1 && out.data[1]==0 && out.data[2]==32767.f/32768.f);clef_audio_samples_free(&out);
    for(size_t i=0;i<n;i++)refuse(p,i,3);
    refuse(p,n,2);refuse(p,n+1,3);
    for(int offset=20;offset<=40;offset+=2) {
        memcpy(q,p,n);q[offset]^=127;refuse(q,n,3);
    }
    n=wave(p,3,4);uint32_t bits[]={0x80000000,0x00000001,0x3f800000,0xc0000000};
    for(int i=0;i<4;i++)u32(p+44+4*i,bits[i]);
    assert(clef_audio_decode(p,n,4,&out,err,sizeof(err)));assert(!memcmp(out.data,bits,sizeof(bits)));clef_audio_samples_free(&out);
    for(int i=0;i<3;i++){u32(p+44,i==0?0x7f800000:i==1?0xff800000:0x7fc01234);refuse(p,n,4);}
    n=wave(p,1,0);refuse(p,n,4);
    n=wave(p,1,1);u16(p+44,1234);
    // Odd ancillary chunks require their pad byte; chunk order need not be fmt/data.
    memcpy(q,p,12);memcpy(q+12,"JUNK",4);u32(q+16,1);q[20]=9;q[21]=0;
    memcpy(q+22,p+36,10);memcpy(q+32,p+12,24);u32(q+4,48);
    assert(clef_audio_decode(q,56,1,&out,err,sizeof(err)));assert(out.data[0]==1234.f/32768.f);clef_audio_samples_free(&out);
    memcpy(q,p,n);memcpy(q+n,p+12,24);u32(q+4,n+24-8);refuse(q,n+24,1);
    memcpy(q,p,n);memcpy(q+n,p+36,10);u32(q+4,n+10-8);refuse(q,n+10,2);
    memcpy(q,p,n);u32(q+40,UINT32_MAX);refuse(q,n,1);
    fail_alloc=1;assert(!clef_audio_decode(p,n,1,&out,err,sizeof(err)));assert(!out.data&&!out.count);fail_alloc=0;
    assert(clef_audio_decode(p,n,1,&out,err,sizeof(err)));clef_audio_samples_free(&out);
    const char *cases[]={"[]","[NaN]","[Infinity]","[-Infinity]","[1e100]","[340282400000000000000000000000000000000]","[true]","[null]","[[0]]","[\"0\"]"};
    for(size_t i=0;i<sizeof(cases)/sizeof(cases[0]);i++) {
        jarena *a=jarena_new();jval *v=json_parse(a,cases[i],strlen(cases[i]),err,sizeof(err));assert(v);
        int before=allocations;assert(!clef_audio_from_json(v,4,&out,err,sizeof(err)));assert(allocations==before);jarena_free(a);
    }
    jarena*a=jarena_new();const char *text="[-1,0,0.5,-0.0,1e-40]";jval*v=json_parse(a,text,strlen(text),err,sizeof(err));assert(v);
    assert(!clef_audio_from_json(v,4,&out,err,sizeof(err)));
    assert(clef_audio_from_json(v,5,&out,err,sizeof(err)) && out.count==5);
    assert(out.data[0]==-1&&out.data[1]==0&&out.data[2]==.5f&&signbit(out.data[3])&&out.data[4]==(float)1e-40);
    clef_audio_samples_free(&out);jarena_free(a);
    // Deterministic mutation sweep exercises structure arithmetic before allocation.
    n=wave(p,1,3);
    for(size_t i=0;i<n;i++)for(int value=0;value<256;value+=17) {
        memcpy(q,p,n);q[i]=value;bool ok=clef_audio_decode(q,n,3,&out,err,sizeof(err));
        if(ok){assert(out.count<=3);clef_audio_samples_free(&out);}else assert(!out.data&&!out.count);
    }
    puts("audio: PCM16/float32 exact samples, JSON, hostile chunks, pre-allocation checks, failures and mutation sweep PASS");
    return 0;
}
