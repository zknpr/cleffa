#include "clef_encoded_video.h"
#include "clef_video.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <mach-o/dyld.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef OUTPUT_CAP
#define OUTPUT_CAP (384u << 20)
#endif
#ifndef DECODE_SECONDS
#define DECODE_SECONDS 30
#endif
static bool error(char *err,size_t n,const char *s) { snprintf(err,n,"encoded video: %s",s);return false; }
static double now(void) { struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9; }
static bool number(const jval *v,double *out) {
    if(!v||(v->type!=J_INT&&v->type!=J_FLOAT))return false;
    *out=v->type==J_FLOAT?v->f:strtod(v->s,NULL);return isfinite(*out)&&*out>0&&*out<=1000;
}
static bool envelope(const jval *v,uint8_t **bytes,size_t *n,char *err,size_t errlen) {
    const jval *payload=v;const char *mime=NULL;size_t mlen=0;
    if(v->type==J_OBJECT) {
        const jval *type=json_get(v,"content_type");payload=json_get(v,"base64");
        if(v->n!=2||!type||type->type!=J_STRING||!payload||payload->type!=J_STRING||
           (payload->len>=5&&!strncasecmp(payload->s,"data:",5)))
            return error(err,errlen,"object requires only content_type and bare base64");
        mime=type->s;mlen=type->len;
    } else if(v->type==J_STRING && v->len>=5 && !strncasecmp(v->s,"data:",5)) {
        const char *end=memchr(v->s+5,';',v->len-5);if(!end)return error(err,errlen,"invalid data URL");
        mime=v->s+5;mlen=(size_t)(end-mime);
    } else return error(err,errlen,"use a video data URL or content_type/base64 object; paths and URLs are not accepted");
    if(!((mlen==9&&!strncasecmp(mime,"video/mp4",9))||(mlen==15&&!strncasecmp(mime,"video/quicktime",15))))
        return error(err,errlen,"content_type must be video/mp4 or video/quicktime");
    if(payload->len>((64u<<20)+2)/3*4+64)return error(err,errlen,"clip exceeds 64 MiB");
    if(!clef_base64_decode(payload->s,payload->len,bytes,n,err,errlen))return false;
    if(*n<8||*n>(64u<<20)|| (memcmp(*bytes+4,"ftyp",4)&&memcmp(*bytes+4,"moov",4)&&memcmp(*bytes+4,"mdat",4)&&memcmp(*bytes+4,"wide",4)&&memcmp(*bytes+4,"free",4))) {
        free(*bytes);*bytes=NULL;return error(err,errlen,"data is not an MP4/MOV container");
    }
    return true;
}
static bool worker(const uint8_t *bytes,size_t n,char **argv,jbuf *out,char *err,size_t errlen) {
    if(errlen)err[0]=0;
    /* tmpfile creates an unlinked private file. No body byte is interpolated into
     * argv or a shell. The worker inherits only its three standard descriptors. */
    FILE *input=tmpfile();if(!input)return error(err,errlen,"cannot create private input");
    if(fwrite(bytes,1,n,input)!=n||fflush(input)||fseek(input,0,SEEK_SET)) {fclose(input);return error(err,errlen,"cannot write private input");}
    int pipes[2]={-1,-1},errors[2]={-1,-1};pid_t pid=-1;bool ok=false;
    posix_spawn_file_actions_t actions;posix_spawnattr_t attr;
    if(pipe(pipes)||pipe(errors))goto done;
    for(int i=0;i<2;i++){if(fcntl(pipes[i],F_SETFD,FD_CLOEXEC)<0||fcntl(errors[i],F_SETFD,FD_CLOEXEC)<0)goto done;}
    int rc=posix_spawn_file_actions_init(&actions);if(rc)goto done;
    rc=posix_spawnattr_init(&attr);if(rc){posix_spawn_file_actions_destroy(&actions);goto done;}
    rc=posix_spawn_file_actions_adddup2(&actions,fileno(input),STDIN_FILENO);
    if(!rc)rc=posix_spawn_file_actions_adddup2(&actions,pipes[1],STDOUT_FILENO);
    if(!rc)rc=posix_spawn_file_actions_adddup2(&actions,errors[1],STDERR_FILENO);
    if(!rc)rc=posix_spawnattr_setflags(&attr,POSIX_SPAWN_CLOEXEC_DEFAULT);
    /* No environment: the worker parses attacker-controlled files and needs none of the
     * server's variables, which can hold deployment secrets. */
    char *empty[]={NULL};
    if(!rc)rc=posix_spawn(&pid,argv[0],&actions,&attr,argv,empty);
    posix_spawnattr_destroy(&attr);posix_spawn_file_actions_destroy(&actions);
    if(rc){snprintf(err,errlen,"encoded video: cannot start clef-media (%s); build it with make media",strerror(rc));goto done;}
    close(pipes[1]);pipes[1]=-1;close(errors[1]);errors[1]=-1;
    double deadline=now()+DECODE_SECONDS;char diagnostics[512]={0};size_t used=0;int status=0;bool exited=false;
    struct pollfd fds[2]={{.fd=pipes[0],.events=POLLIN},{.fd=errors[0],.events=POLLIN}};
    while(!exited||fds[0].fd>=0||fds[1].fd>=0) {
        if(now()>=deadline){error(err,errlen,"decoder exceeded 30 second deadline");goto done;}
        int polled=poll(fds,2,100);if(polled<0&&errno!=EINTR){error(err,errlen,"decoder pipe poll failed");goto done;}
        for(int i=0;i<2;i++)if(fds[i].fd>=0 && (fds[i].revents&(POLLIN|POLLHUP|POLLERR))) {
            char chunk[65536];ssize_t got=read(fds[i].fd,chunk,sizeof(chunk));
            if(got<0){if(errno==EINTR)continue;error(err,errlen,"decoder pipe read failed");goto done;}
            if(!got){fds[i].fd=-1;continue;}
            if(i==0) {
                if((size_t)got>OUTPUT_CAP-out->len){error(err,errlen,"decoded output exceeds 384 MiB");goto done;}
                jbuf_put(out,chunk,(size_t)got);if(out->oom){error(err,errlen,"out of memory");goto done;}
            } else {size_t count=(size_t)got;if(count>sizeof(diagnostics)-1-used)count=sizeof(diagnostics)-1-used;memcpy(diagnostics+used,chunk,count);used+=count;}
        }
        if(!exited) {
            pid_t result=waitpid(pid,&status,WNOHANG);
            if(result==pid){exited=true;pid=-1;}
            else if(result<0&&errno!=EINTR){error(err,errlen,"decoder wait failed");goto done;}
        }
    }
    if(!WIFEXITED(status)||WEXITSTATUS(status)) {
        /* Codec diagnostics are untrusted. Keep the HTTP error on one line. */
        for(size_t i=0;i<used;i++)if((unsigned char)diagnostics[i]<32)diagnostics[i]=' ';
        snprintf(err,errlen,"encoded video: decoder failed: %.400s",used?diagnostics:"no diagnostics");goto done;
    }
    ok=true;
done:
    if(pid>0){kill(pid,SIGKILL);while(waitpid(pid,NULL,0)<0&&errno==EINTR){}}
    for(int i=0;i<2;i++){if(pipes[i]>=0)close(pipes[i]);if(errors[i]>=0)close(errors[i]);}
    fclose(input);if(!ok&&errlen&&!err[0])error(err,errlen,"cannot initialize decoder process");return ok;
}
void clef_video_request_free(clef_video_request *request) {
    free(request->members);jarena_free(request->arena);jbuf_free(&request->storage);memset(request,0,sizeof(*request));
}
bool clef_video_expand(const jval *request,clef_encode_opts opts,clef_video_request *out,char *err,size_t errlen) {
    memset(out,0,sizeof(*out));if(!request||request->type!=J_OBJECT)return true;
    const jval *videos=json_get(request,"videos");if(!videos||videos->type!=J_ARRAY||!videos->n)return true;
    size_t encoded=0;for(size_t i=0;i<videos->n;i++)encoded+=!json_get(videos->items[i],"frames");
    if(!encoded)return true;
    // Refused before the envelope is decoded or a worker is spawned; see allow_encoded_video.
    if(!opts.vision.allow_encoded_video)return error(err,errlen,"MP4/MOV uploads are disabled (clef-server --allow-encoded-video enables them)");
    if(encoded!=videos->n)return error(err,errlen,"use encoded clips or frame arrays consistently in one request");
    if(!opts.vision.video_token_id)return error(err,errlen,"model has no video tower");
    if(videos->n>32||(opts.vision.max_videos>0&&videos->n>(size_t)opts.vision.max_videos))return error(err,errlen,"too many videos");
    const bool omni=opts.model.backbone==CLEF_OMNI;const jval *mk=json_get(request,"media_kwargs");
    if(mk&&mk->type==J_NULL)mk=NULL;  // absent, as in encode_media; the expansion replaces it
    if(mk&&mk->type!=J_OBJECT)return error(err,errlen,"media_kwargs must be an object");
    const jval *vk=mk?json_get(mk,"videos_kwargs"):NULL;
    double fps=2;int requested=0;bool sample=!omni,explicit_fps=false;
    if(vk) {
        if(vk->type!=J_OBJECT)return error(err,errlen,"videos_kwargs must be an object");
        for(size_t i=0;i<vk->n;i++) {
            const jmember *m=vk->members+i;const jval *v=m->val;
            if(m->klen==3&&!memcmp(m->key,"fps",3)) {explicit_fps=v->type!=J_NULL;if(explicit_fps&&!number(v,&fps))return error(err,errlen,"invalid sampling fps");}
            else if(m->klen==10&&!memcmp(m->key,"num_frames",10)) {double value;if(v->type!=J_INT||!number(v,&value)||value>768)return error(err,errlen,"invalid num_frames");requested=(int)value;}
            else if(m->klen==16&&!memcmp(m->key,"do_sample_frames",16)) {if(v->type!=J_FALSE&&v->type!=J_TRUE)return error(err,errlen,"do_sample_frames must be boolean");sample=v->type==J_TRUE;}
            else if(!(m->klen==4&&!memcmp(m->key,"size",4)) && !(omni&&m->klen==18&&!memcmp(m->key,"use_audio_in_video",18)))return error(err,errlen,"unsupported videos_kwargs");
        }
    }
    if(omni&&(sample||requested||(vk&&!explicit_fps)||fps!=2))return error(err,errlen,"Omni requires fps=2 and do_sample_frames=false");
    if(!omni&&((requested&&explicit_fps)||(!sample&&(requested||explicit_fps))))return error(err,errlen,"choose fps or num_frames, neither with do_sample_frames=false");
    char executable[PATH_MAX],resolved[PATH_MAX];uint32_t length=sizeof(executable);
    if(_NSGetExecutablePath(executable,&length)||!realpath(executable,resolved))return error(err,errlen,"cannot locate decoder sibling");
    char *slash=strrchr(resolved,'/');if(!slash||strlen(resolved)+12>=sizeof(resolved))return error(err,errlen,"decoder path too long");strcpy(slash+1,"clef-media");
    char rate[32],count[16],frames[16],pixels[32],samples[32];
    snprintf(rate,sizeof(rate),"%.17g",fps);snprintf(count,sizeof(count),"%d",requested);
    snprintf(frames,sizeof(frames),"%d",opts.vision.max_video_frames>0&&opts.vision.max_video_frames<768?opts.vision.max_video_frames:768);
    snprintf(pixels,sizeof(pixels),"%ld",opts.vision.max_image_pixels>0&&opts.vision.max_image_pixels<16777216?opts.vision.max_image_pixels:16777216L);
    size_t cap=opts.audio.max_samples?opts.audio.max_samples:9600000;if(cap>9600000)cap=9600000;
    snprintf(samples,sizeof(samples),"%zu",cap);
    char *argv[]={resolved,omni?"omni":"dense",rate,count,sample?"1":"0",frames,pixels,samples,NULL};
    jbuf result={0};jbuf_puts(&result,"{\"videos\":[");out->arena=jarena_new();if(!out->arena)return error(err,errlen,"out of memory");
    bool ok=false;int tracks=0;
    for(size_t i=0;i<videos->n;i++) {
        uint8_t *bytes=NULL;size_t n=0;jbuf decoded={0};
        if(!envelope(videos->items[i],&bytes,&n,err,errlen))goto done;
        bool success=worker(bytes,n,argv,&decoded,err,errlen);free(bytes);
        if(!success){jbuf_free(&decoded);goto done;}
        if(decoded.len>OUTPUT_CAP-result.len){jbuf_free(&decoded);error(err,errlen,"total decoded output exceeds 384 MiB");goto done;}
        jval *video=json_parse(out->arena,decoded.p,decoded.len,err,errlen);
        if(!video){jbuf_free(&decoded);goto done;}
        tracks+=json_get(video,"audio")!=NULL;
        if(i)jbuf_puts(&result,",");jbuf_put(&result,decoded.p,decoded.len);jbuf_free(&decoded);
    }
    jbuf_puts(&result,"],\"media_kwargs\":{");
    if(mk)for(size_t i=0;i<mk->n;i++) {
        const jmember *m=mk->members+i;if(m->klen==13&&!memcmp(m->key,"videos_kwargs",13))continue;
        json_put_string(&result,m->key,m->klen);jbuf_puts(&result,":");json_dump(&result,m->val,false);jbuf_puts(&result,",");
    }
    jbuf_puts(&result,"\"videos_kwargs\":");
    if(omni&&vk)json_dump(&result,vk,false);
    else {
        jbuf_puts(&result,"{");
        if(vk){const jval *size=json_get(vk,"size");if(size){jbuf_puts(&result,"\"size\":");json_dump(&result,size,false);jbuf_puts(&result,",");}}
        jbuf_puts(&result,"\"do_sample_frames\":false");
        if(omni){jbuf_puts(&result,",\"fps\":2,\"use_audio_in_video\":");jbuf_puts(&result,tracks?"true":"false");}
        jbuf_puts(&result,"}");
    }
    jbuf_puts(&result,"}}");if(result.oom){error(err,errlen,"out of memory");goto done;}
    jval *media=json_parse(out->arena,result.p,result.len,err,errlen);if(!media)goto done;
    out->members=calloc(request->n+1,sizeof(*out->members));if(!out->members){error(err,errlen,"out of memory");goto done;}
    out->request=*request;out->request.members=out->members;memcpy(out->members,request->members,request->n*sizeof(*out->members));bool had_mk=false;
    for(size_t i=0;i<request->n;i++) {
        jmember *m=out->members+i;
        if(m->klen==6&&!memcmp(m->key,"videos",6))m->val=json_get(media,"videos");
        if(m->klen==12&&!memcmp(m->key,"media_kwargs",12)){m->val=json_get(media,"media_kwargs");had_mk=true;}
    }
    if(!had_mk)out->members[out->request.n++]=(jmember){"media_kwargs",12,json_get(media,"media_kwargs")};
    /* Unescaped JSON strings borrow the input buffer. Retain it until media
     * encoding finishes, while request/question strings keep their old owner. */
    out->storage=result;result=(jbuf){0};ok=true;
done:
    jbuf_free(&result);if(!ok)clef_video_request_free(out);return ok;
}
