// Offline tower qualification, using the same model validation as inference.
#include "../clef.c"
#include "../clef_metal_internal.h"
#include <errno.h>

static bool save(void *context,const char *stage,int first,const float *values,int rows,int columns,char *err,size_t errlen) {
    char path[4096];
    int count=snprintf(path,sizeof(path),"%s/%s.%d.f32",(const char *)context,stage,first);
    if(count<0 || (size_t)count>=sizeof(path)){snprintf(err,errlen,"audio capture path too long");return false;}
    FILE *f=fopen(path,"wb");
    if(!f){snprintf(err,errlen,"cannot open audio capture: %s",strerror(errno));return false;}
    bool ok=fwrite(values,4,(size_t)rows*columns,f)==(size_t)rows*columns;
    if(fclose(f))ok=false;
    if(!ok)snprintf(err,errlen,"cannot write audio capture");
    return ok;
}
int main(int argc,char **argv) { @autoreleasepool {
    if(argc<6){fprintf(stderr,"usage: audio-tower GGUF OUTPUT PADDED_FRAMES MEL.f32 FRAMES [MEL.f32 FRAMES ...]\n");return 2;}
    if((argc-4)%2)return 2;
    int n=(argc-4)/2,padded=atoi(argv[3]),rc=1;
    if(n>4 || padded<1 || padded>30000)return 2;
    clef_engine *e=calloc(1,sizeof(*e));clef_audio_features *features=calloc(n,sizeof(*features));
    char err[512]="audio tower input failure";
    if(!e || !features)goto done;
    if(!gguf_open(&e->gguf,argv[1],err,sizeof(err)))goto done;
    if(!load_config(&e->gguf,&e->cfg,err,sizeof(err)))goto done;
    if(e->cfg.audio.H!=1280 || !(e->cfg.model.capabilities&CLEF_CAP_AUDIO))goto done;
    if(!bind_weights(&e->gguf,&e->cfg,&e->w,err,sizeof(err)))goto done;
    e->gpu=clef_gpu_open(e,err,sizeof(err));if(!e->gpu)goto done;
    for(int i=0;i<n;i++) {
        int frames=atoi(argv[5+2*i]);if(frames<1 || frames>padded)goto done;
        features[i]=(clef_audio_features){.frames=frames,.padded_frames=padded,.n_tokens=13*(frames/100)+(frames%100+7)/8};
        features[i].mel=malloc((size_t)128*padded*4);if(!features[i].mel)goto done;
        FILE *f=fopen(argv[4+2*i],"rb");if(!f)goto done;
        bool ok=fread(features[i].mel,4,(size_t)128*padded,f)==(size_t)128*padded && fgetc(f)==EOF && !ferror(f);
        if(fclose(f))ok=false;
        if(!ok)goto done;
        // The fixture holds the reference's padded [128][padded]; the tower takes own frames.
        for(int m=1;m<128;m++)memmove(features[i].mel+(size_t)m*frames,features[i].mel+(size_t)m*padded,(size_t)frames*4);
    }
    {
        id<MTLBuffer> out=nil;
        if(!clef_audio_gpu_run(e->gpu,e,features,n,&out,save,argv[2],err,sizeof(err)))goto done;
        if(!out)goto done;
    }
    rc=0;
done:
    if(rc)fprintf(stderr,"%s\n",err);
    if(features){for(int i=0;i<n;i++)clef_audio_features_free(features+i);free(features);}
    if(e){clef_gpu_close(e->gpu);gguf_close(&e->gguf);free(e);}
    return rc;
} }
