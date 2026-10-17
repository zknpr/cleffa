/* A decoder changing layout must fail before swr reads any sample planes. */
#define main clef_media_worker_main
#include "../clef_media.c"
#undef main
#include <assert.h>
#include <sys/wait.h>
#include <unistd.h>
int main(void) {
    for(int mode=0;mode<3;mode++) {
        pid_t child=fork();assert(child>=0);
        if(!child) {
            SwrContext *swr=NULL;AVChannelLayout input=AV_CHANNEL_LAYOUT_STEREO,output=AV_CHANNEL_LAYOUT_MONO;
            check(swr_alloc_set_opts2(&swr,&output,AV_SAMPLE_FMT_FLT,16000,&input,AV_SAMPLE_FMT_FLTP,16000,0,NULL),"setup");
            check(swr_init(swr),"init");
            AVFrame *frame=av_frame_alloc();assert(frame);
            frame->format=mode==0?AV_SAMPLE_FMT_S16:AV_SAMPLE_FMT_FLTP;
            frame->sample_rate=mode==1?8000:16000;
            frame->ch_layout=mode==2?output:input;
            frame->nb_samples=32;
            /* No planes allocated: reading them instead of rejecting metadata
             * is a sanitizer failure or signal, never the expected exit(1). */
            float samples[64];size_t count=0;
            audio_samples(swr,frame,samples,&count,64);
            _exit(3);
        }
        int status;assert(waitpid(child,&status,0)==child);
        assert(WIFEXITED(status)&&WEXITSTATUS(status)==1);
    }
    puts("audio format transitions rejected before sample reads");return 0;
}
