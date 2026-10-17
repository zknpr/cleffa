#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include "../clef_audio.h"

int main(int argc,char **argv) {
    if(argc<2 || argc>5)return 2;
    clef_audio_features clips[4]={{0}};clef_audio_layout layout={0};float dummy=0;size_t longest=0;char err[256];
    for(int i=1;i<argc;i++) {
        char *end;errno=0;long count=strtol(argv[i],&end,10);
        if(errno || !*argv[i] || *end || count<1 || count>1000000)return 2;
        clips[i-1]=(clef_audio_features){.mel=&dummy,.frames=(size_t)count,.n_tokens=(int)(13*(count/100)+(count%100+7)/8)};
        if((size_t)count>longest)longest=(size_t)count;
    }
    for(int i=0;i<argc-1;i++)clips[i].padded_frames=longest;
    if(!clef_audio_layout_make(clips,argc-1,&layout,err,sizeof(err))){fprintf(stderr,"%s\n",err);return 1;}
    printf("{\"max_chunk\":%d,\"chunks\":[",layout.max_chunk_frames);
    for(int i=0;i<layout.n_chunks;i++) {
        clef_audio_chunk c=layout.chunks[i];
        printf("%s[%d,%d,%d,%d]",i?",":"",c.clip,c.frame,c.frames,c.first_token);
    }
    const int32_t *arrays[]={layout.token_position,layout.window_start,layout.window_end};
    const char *names[]={"positions","starts","ends"};
    for(int a=0;a<3;a++) {
        printf("],\"%s\":[",names[a]);
        for(int i=0;i<layout.tokens;i++)printf("%s%d",i?",":"",arrays[a][i]);
    }
    puts("]}");clef_audio_layout_free(&layout);return 0;
}
