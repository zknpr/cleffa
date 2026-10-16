/* Optional codec worker. The engine passes only an owned byte stream on stdin;
 * custom AVIO cannot open a URL or an external MOV data reference. A separate
 * process bounds codec crashes, CPU time and decoded output. */
#include "clef_json.h"
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

#define INPUT_CAP (64u << 20)
#define RAW_CAP (256u << 20)
static void die(const char *s) { fprintf(stderr,"video decoder: %s\n",s); exit(1); }
static void check(int rc, const char *s) { if(rc<0)die(s); }
static void base64(jbuf *b,const uint8_t *p,size_t n) {
    static const char alphabet[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for(size_t i=0;i<n;i+=3) {
        uint32_t v=(uint32_t)p[i]<<16;
        if(i+1<n)v|=(uint32_t)p[i+1]<<8;
        if(i+2<n)v|=p[i+2];
        char out[4]={alphabet[v>>18],alphabet[(v>>12)&63],i+1<n?alphabet[(v>>6)&63]:'=',i+2<n?alphabet[v&63]:'='};
        jbuf_put(b,out,4);
    }
    if(b->oom)die("out of memory");
}
typedef struct { uint8_t *data; size_t n,at; } source;
static int read_bytes(void *opaque,uint8_t *out,int n) {
    source *s=opaque;
    if(s->at==s->n)return AVERROR_EOF;
    size_t count=s->n-s->at;if(count>(size_t)n)count=(size_t)n;
    memcpy(out,s->data+s->at,count);s->at+=count;return (int)count;
}
static int64_t seek_bytes(void *opaque,int64_t offset,int whence) {
    source *s=opaque;
    if(whence==AVSEEK_SIZE)return (int64_t)s->n;
    int64_t base=whence==SEEK_SET?0:whence==SEEK_CUR?(int64_t)s->at:whence==SEEK_END?(int64_t)s->n:-1;
    if(base<0 || offset < -base || offset>(int64_t)s->n-base)return AVERROR(EINVAL);
    s->at=(size_t)(base+offset);return (int64_t)s->at;
}
static int deny_open(AVFormatContext *s,AVIOContext **pb,const char *url,int flags,AVDictionary **options) {
    return AVERROR(EACCES);
}
static AVFormatContext *open_source(source *src) {
    src->at=0;
    AVFormatContext *f=avformat_alloc_context();
    uint8_t *buffer=av_malloc(32768);
    if(!f || !buffer)die("out of memory");
    f->pb=avio_alloc_context(buffer,32768,0,src,read_bytes,NULL,seek_bytes);
    if(!f->pb)die("out of memory");
    f->flags|=AVFMT_FLAG_CUSTOM_IO;f->io_open=deny_open;
    AVDictionary *opts=NULL;
    av_dict_set(&opts,"enable_drefs","0",0);av_dict_set(&opts,"use_absolute_path","0",0);
    av_dict_set(&opts,"protocol_whitelist","",0);av_dict_set(&opts,"max_streams","16",0);
    av_dict_set(&opts,"probesize","1048576",0);av_dict_set(&opts,"analyzeduration","1000000",0);
    check(avformat_open_input(&f,NULL,av_find_input_format("mov"),&opts),"invalid MP4/MOV");av_dict_free(&opts);
    /* MOV carries the required stream parameters in its header. Do not call
     * find_stream_info: it may decode probe frames before our pixel/codec checks. */
    return f;
}
static void close_source(AVFormatContext **f) {
    AVIOContext *io=(*f)->pb;avformat_close_input(f);av_freep(&io->buffer);avio_context_free(&io);
}
static AVCodecContext *decoder(AVStream *stream,long pixels) {
    const AVCodec *codec=avcodec_find_decoder(stream->codecpar->codec_id);
    if(!codec)die("unsupported codec");
    AVCodecContext *ctx=avcodec_alloc_context3(codec);if(!ctx)die("out of memory");
    check(avcodec_parameters_to_context(ctx,stream->codecpar),"invalid codec parameters");
    ctx->thread_count=1;ctx->max_pixels=pixels;ctx->max_samples=16000000;ctx->err_recognition=AV_EF_EXPLODE;
    check(avcodec_open2(ctx,codec,NULL),"cannot open codec");return ctx;
}
static jbuf png(AVFrame *src,int w,int h) {
    AVFrame *rgb=av_frame_alloc();if(!rgb)die("out of memory");
    check(av_frame_copy_props(rgb,src),"cannot copy frame metadata");
    rgb->format=AV_PIX_FMT_RGB24;rgb->width=w;rgb->height=h;
    /* Match PyAV 17's dynamic scaler: chroma location matters on downscales.
     * Transfer/primaries conversion is opt-in there, so neutralize those tags. */
    src->color_trc=rgb->color_trc=AVCOL_TRC_UNSPECIFIED;
    src->color_primaries=rgb->color_primaries=AVCOL_PRI_UNSPECIFIED;
    if(src->colorspace==5)src->colorspace=rgb->colorspace=AVCOL_SPC_SMPTE170M;
    struct SwsContext *scale=sws_alloc_context();
    if(!scale)die("cannot create RGB scaler");
    scale->flags=SWS_BILINEAR;scale->threads=1;
    check(av_frame_get_buffer(rgb,32),"cannot allocate RGB frame");
    check(sws_scale_frame(scale,rgb,src),"RGB conversion failed");sws_freeContext(scale);
    const AVCodec *codec=avcodec_find_encoder(AV_CODEC_ID_PNG);
    AVCodecContext *ctx=avcodec_alloc_context3(codec);if(!ctx)die("out of memory");
    ctx->width=w;ctx->height=h;ctx->pix_fmt=AV_PIX_FMT_RGB24;ctx->time_base=(AVRational){1,2};ctx->thread_count=1;
    ctx->compression_level=1;
    check(avcodec_open2(ctx,codec,NULL),"cannot open PNG encoder");
    check(avcodec_send_frame(ctx,rgb),"cannot encode frame");
    AVPacket *packet=av_packet_alloc();if(!packet)die("out of memory");
    check(avcodec_receive_packet(ctx,packet),"cannot receive PNG");
    jbuf out={0};jbuf_puts(&out,"\"data:image/png;base64,");base64(&out,packet->data,(size_t)packet->size);jbuf_puts(&out,"\"");
    av_packet_free(&packet);avcodec_free_context(&ctx);av_frame_free(&rgb);return out;
}
static void output(const char *p,size_t n) { if(fwrite(p,1,n,stdout)!=n)die("output write failed"); }
static void put(const char *s) { output(s,strlen(s)); }
static void put32(uint8_t *p,uint32_t v) { for(int i=0;i<4;i++)p[i]=(uint8_t)(v>>(8*i)); }
static void audio_samples(SwrContext *swr,AVFrame *f,float *samples,size_t *count,size_t cap) {
    if(f) {
        int64_t rate=0;enum AVSampleFormat format;AVChannelLayout layout={0};
        check(av_opt_get_int(swr,"in_sample_rate",0,&rate),"cannot read resampler rate");
        check(av_opt_get_sample_fmt(swr,"in_sample_fmt",0,&format),"cannot read resampler format");
        check(av_opt_get_chlayout(swr,"in_chlayout",0,&layout),"cannot read resampler layout");
        bool match=f->sample_rate==rate && f->format==format && !av_channel_layout_compare(&f->ch_layout,&layout);
        av_channel_layout_uninit(&layout);
        /* swr_convert receives raw channel pointers, not frame metadata. A
         * changing layout must fail before it can read nonexistent planes. */
        if(!match)die("audio frame format, rate or layout changed");
    }
    int available=swr_get_out_samples(swr,f?f->nb_samples:0);check(available,"invalid audio resampler size");
    if(available<1)return;
    /* swr's upper bound includes delay; leave space for that bound without
     * treating unused capacity as decoded samples. Actual count is checked. */
    float *scratch=av_malloc((size_t)available*sizeof(float));if(!scratch)die("out of memory");
    uint8_t *out=(uint8_t *)scratch;
    int n=swr_convert(swr,&out,available,f?(const uint8_t **)f->extended_data:NULL,f?f->nb_samples:0);
    check(n,"audio resampling failed");
    if((size_t)n>cap-*count)die("audio sample limit exceeded");
    for(int i=0;i<n;i++)if(!isfinite(scratch[i]))die("nonfinite audio sample");
    memcpy(samples+*count,scratch,(size_t)n*sizeof(float));*count+=(size_t)n;av_free(scratch);
}
static void soundtrack(source *src,size_t cap,long pixels) {
    AVFormatContext *f=open_source(src);int index=-1;
    for(unsigned i=0;i<f->nb_streams;i++)if(f->streams[i]->codecpar->codec_type==AVMEDIA_TYPE_AUDIO) {
        if(index>=0)die("exactly one audio track is supported");index=(int)i;
    }
    if(index<0){close_source(&f);return;}
    AVCodecContext *ctx=decoder(f->streams[index],pixels);
    SwrContext *swr=NULL;AVChannelLayout mono=AV_CHANNEL_LAYOUT_MONO;
    float *samples=malloc(cap*sizeof(float));if(!samples)die("out of memory");size_t count=0;
    AVPacket *packet=av_packet_alloc();AVFrame *frame=av_frame_alloc();if(!packet||!frame)die("out of memory");
    bool end=false;
    while(!end) {
        int rc=av_read_frame(f,packet);end=rc==AVERROR_EOF;
        if(!end)check(rc,"audio demux failed");
        if(!end && packet->stream_index!=index){av_packet_unref(packet);continue;}
        check(avcodec_send_packet(ctx,end?NULL:packet),"audio packet failed");av_packet_unref(packet);
        while((rc=avcodec_receive_frame(ctx,frame))>=0) {
            if(!swr) {
                check(swr_alloc_set_opts2(&swr,&mono,AV_SAMPLE_FMT_FLT,16000,&frame->ch_layout,frame->format,frame->sample_rate,0,NULL),"audio layout failed");
                check(swr_init(swr),"audio resampler failed");
            }
            audio_samples(swr,frame,samples,&count,cap);av_frame_unref(frame);
        }
        if(rc!=AVERROR_EOF && rc!=AVERROR(EAGAIN))check(rc,"audio decoding failed");
    }
    if(swr)audio_samples(swr,NULL,samples,&count,cap);
    if(!count)die("audio track has no samples");
    size_t size=44+count*4;uint8_t *wav=malloc(size);if(!wav)die("out of memory");
    memcpy(wav,"RIFF\0\0\0\0WAVEfmt \20\0\0\0\3\0\1\0\200\76\0\0\0\372\0\0\4\0\40\0data",40);
    put32(wav+4,(uint32_t)(size-8));put32(wav+40,(uint32_t)(count*4));memcpy(wav+44,samples,count*4);
    jbuf encoded={0};base64(&encoded,wav,size);put(",\"audio\":\"data:audio/wav;base64,");output(encoded.p,encoded.len);put("\"");
    jbuf_free(&encoded);free(wav);free(samples);swr_free(&swr);av_frame_free(&frame);av_packet_free(&packet);avcodec_free_context(&ctx);close_source(&f);
}
int main(int argc,char **argv) {
    if(argc!=8)die("invalid worker arguments");
    struct rlimit cpu={30,30},core={0,0};if(setrlimit(RLIMIT_CPU,&cpu)||setrlimit(RLIMIT_CORE,&core))die("cannot set resource limits");
    av_log_set_level(AV_LOG_ERROR);av_max_alloc(64u<<20);
    bool omni=!strcmp(argv[1],"omni"),sample=atoi(argv[4])!=0;
    double target=strtod(argv[2],NULL);int requested=atoi(argv[3]),max_frames=atoi(argv[5]);long pixels=strtol(argv[6],NULL,10);size_t audio_cap=strtoul(argv[7],NULL,10);
    if(!isfinite(target)||target<=0||target>1000||requested<0||requested>768||max_frames<1||max_frames>768||pixels<1||pixels>16777216||audio_cap<1||audio_cap>9600000)die("invalid worker limits");
    source src={.data=malloc(INPUT_CAP+1)};if(!src.data)die("out of memory");
    src.n=fread(src.data,1,INPUT_CAP+1,stdin);if(ferror(stdin)||src.n<8||src.n>INPUT_CAP)die("encoded clip must be 8 bytes to 64 MiB");
    AVFormatContext *f=open_source(&src);int index=-1;
    for(unsigned i=0;i<f->nb_streams;i++)if(f->streams[i]->codecpar->codec_type==AVMEDIA_TYPE_VIDEO) {
        if(index>=0)die("exactly one video track is required");index=(int)i;
    }
    if(index<0)die("video track missing");AVStream *stream=f->streams[index];AVCodecParameters *p=stream->codecpar;
    if(p->codec_id!=AV_CODEC_ID_H264 && p->codec_id!=AV_CODEC_ID_HEVC && p->codec_id!=AV_CODEC_ID_PRORES && p->codec_id!=AV_CODEC_ID_MJPEG)die("supported codecs: H.264, HEVC, ProRes, MJPEG");
    double fps=av_q2d(stream->avg_frame_rate);int64_t total=stream->nb_frames;
    if(total<1||total>18000||!isfinite(fps)||fps<=0||fps>1000||total/fps>600||p->width<32||p->height<32||p->width>16384||p->height>16384||(int64_t)p->width*p->height>pixels)die("invalid or excessive frame count, duration or dimensions");
    int desired=sample?(requested?requested:(int)fmin(fmin(fmax(trunc(total/fps*target),4),768),total)):(int)total;
    int indices[768];
    if(!omni) {
        if(desired<1||desired>total||desired>max_frames)die("sampled frame limit exceeded");
        for(int i=0;i<desired;i++)indices[i]=sample&&desired>1?(int)rint(i*((total-1.0)/(desired-1))):i;
        if(sample&&desired>1)indices[desired-1]=(int)total-1;
    }
    AVCodecContext *ctx=decoder(stream,pixels);AVPacket *packet=av_packet_alloc();AVFrame *frame=av_frame_alloc();if(!packet||!frame)die("out of memory");
    int decoded=0,count=0;size_t raw=0;bool end=false;jbuf last={0};put("{\"frames\":[");
    while(!end) {
        int rc=av_read_frame(f,packet);end=rc==AVERROR_EOF;if(!end)check(rc,"video demux failed");
        if(!end&&packet->stream_index!=index){av_packet_unref(packet);continue;}
        check(avcodec_send_packet(ctx,end?NULL:packet),"video packet failed");av_packet_unref(packet);
        while((rc=avcodec_receive_frame(ctx,frame))>=0) {
            if(decoded>=total || frame->width!=p->width || frame->height!=p->height)die("decoded geometry/count disagrees with metadata");
            bool take=omni?(frame->pts!=AV_NOPTS_VALUE && frame->pts*av_q2d(stream->time_base)+1e-6>=count/2.0):(count<desired&&decoded==indices[count]);decoded++;
            if(take) {
                if(count>=max_frames)die("sampled frame limit exceeded");
                double scale=omni?fmin(1,sqrt(262144.0/(frame->width*(double)frame->height))):1;
                int w=omni?(int)fmax(2,rint(frame->width*scale/2)*2):frame->width,h=omni?(int)fmax(2,rint(frame->height*scale/2)*2):frame->height;
                size_t bytes=(size_t)w*h*3;if(bytes>RAW_CAP-raw)die("sampled RGB exceeds 256 MiB");raw+=bytes;
                jbuf_free(&last);last=png(frame,w,h);if(count)put(",");output(last.p,last.len);count++;
            }
            av_frame_unref(frame);
        }
        if(rc!=AVERROR_EOF&&rc!=AVERROR(EAGAIN))check(rc,"video decoding failed");
    }
    if(decoded!=total||!count||(!omni&&count!=desired))die("decoded frame count disagrees with metadata");
    if(omni&&count%2){if(count==max_frames)die("duplicated final frame exceeds limit");put(",");output(last.p,last.len);count++;}
    jbuf_free(&last);put("]");char number[128];snprintf(number,sizeof(number),",\"fps\":%.17g",omni?2:fps);put(number);
    if(!omni) {
        snprintf(number,sizeof(number),",\"total_num_frames\":%lld,\"frame_indices\":[",(long long)total);put(number);
        for(int i=0;i<count;i++){snprintf(number,sizeof(number),"%s%d",i?",":"",indices[i]);put(number);}put("]");
    }
    av_frame_free(&frame);av_packet_free(&packet);avcodec_free_context(&ctx);close_source(&f);
    if(omni)soundtrack(&src,audio_cap,pixels);
    put("}\n");free(src.data);if(fflush(stdout))die("output flush failed");return 0;
}
