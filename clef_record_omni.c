#include "clef_record_omni.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

static bool error(char *err, size_t n, const char *message) {
    snprintf(err, n, "Omni media: %s", message);
    return false;
}

static clef_audio_ref *video_audio(clef_record *record,int group) {
    if(group)for(int a=0;a<record->n_audio;a++)
        if(record->audio[a].video_index==group-1)return record->audio+a;
    return NULL;
}

static bool append_audio_token(clef_audio_ref *clip,int index,int32_t id,clef_tokens *prefix) {
    if(prefix->len>INT32_MAX)return false;
    clip->token_rows[index]=(int32_t)prefix->len;
    return clef_tokens_push(prefix,id);
}

bool clef_omni_append_visual_tokens(const clef_tokenizer *tok, clef_record *record,
                                     const clef_encode_opts *opts, clef_tokens *prefix,
                                     char *err, size_t errlen) {
    const clef_vision_opts *vision=&opts->vision;
    clef_tokens newline = {0};
    if (!clef_tok_encode_ex(tok, "\n", 1, SIZE_MAX, false, &newline)) {
        clef_tokens_free(&newline);
        return error(err, errlen, "cannot encode newline");
    }
    if (newline.len != 1) {
        clef_tokens_free(&newline);
        return error(err, errlen, "newline must occupy one token");
    }
    bool ok = true;
    bool audio_done=false;
    for (int i = 0; ok && (i < record->n_images || !audio_done);) {
        if(!audio_done && (i==record->n_images || record->images[i].video_group)) {
            for(int a=0;ok && a<record->n_audio;a++) {
                clef_audio_ref *clip=record->audio+a;
                if(clip->video_index>=0)continue;
                ok=clef_tokens_push(prefix,opts->audio.start_token_id);
                for(int t=0;ok && t<clip->feat.n_tokens;t++) {
                    ok=append_audio_token(clip,t,opts->audio.token_id,prefix);
                }
                ok=ok && clef_tokens_push(prefix,opts->audio.end_token_id) && clef_tokens_push(prefix,newline.ids[0]);
            }
            audio_done=true;
        }
        if(!ok || i==record->n_images)break;
        const int group = record->images[i].video_group;
        clef_audio_ref *track=video_audio(record,group);
        int audio_index=0,frame=0;
        ok = clef_tokens_push(prefix, vision->start_token_id);
        if(track)ok=ok && clef_tokens_push(prefix,opts->audio.start_token_id);
        do {
            // Stable temporal merge: every visual row at a tied timestamp
            // precedes its audio row, as in the pinned processor.
            while(ok && track && audio_index<track->feat.n_tokens && audio_index<(int64_t)frame*13)
                ok=append_audio_token(track,audio_index++,opts->audio.token_id,prefix);
            clef_image_ref *image = &record->images[i];
            if (prefix->len > INT32_MAX) { ok = false; break; }
            image->tok_start = (int32_t)prefix->len;
            for (int j = 0; ok && j < image->pt.n_tokens; j++)
                ok = clef_tokens_push(prefix, group ? vision->video_token_id : vision->image_token_id);
            i++;frame++;
        } while (ok && group && i < record->n_images && record->images[i].video_group == group);
        while(ok && track && audio_index<track->feat.n_tokens)
            ok=append_audio_token(track,audio_index++,opts->audio.token_id,prefix);
        if(track)ok=ok && clef_tokens_push(prefix,opts->audio.end_token_id);
        ok = ok && clef_tokens_push(prefix, vision->end_token_id) && clef_tokens_push(prefix, newline.ids[0]);
    }
    clef_tokens_free(&newline);
    return ok || error(err, errlen, "cannot append visual tokens");
}

bool clef_audio_opts_load(const gguf_file *f,clef_audio_opts *audio,char *err,size_t errlen) {
    audio->token_id=audio->start_token_id=audio->end_token_id=0;
    clef_model_info model;
    if(!clef_model_info_load(f,&model,err,errlen))return false;
    if(!(model.capabilities&CLEF_CAP_AUDIO))return true;
    clef_audio_config config;
    if(!clef_audio_config_load(f,&config,err,errlen))return false;
    const char *keys[]={"clef.audio.audio_token_id","clef.audio.audio_start_token_id","clef.audio.audio_end_token_id"};
    uint32_t ids[3],vocab;
    if(!gguf_get_u32(f,"clef.vocab_size",&vocab))return error(err,errlen,"missing vocabulary size");
    for(int i=0;i<3;i++) {
        if(!gguf_get_u32(f,keys[i],ids+i) || !ids[i] || ids[i]>=vocab || ids[i]>INT32_MAX)
            return error(err,errlen,"invalid audio token ID");
        for(int j=0;j<i;j++)if(ids[j]==ids[i])return error(err,errlen,"audio token IDs must be distinct");
        const char *visual[]={"clef.vision.image_token_id","clef.vision.video_token_id","clef.vision.start_token_id","clef.vision.end_token_id"};
        for(size_t j=0;j<sizeof(visual)/sizeof(visual[0]);j++) {
            uint32_t other;
            if(!gguf_get_u32(f,visual[j],&other) || ids[i]==other)return error(err,errlen,"audio and visual token IDs must be distinct");
        }
    }
    audio->token_id=ids[0];audio->start_token_id=ids[1];audio->end_token_id=ids[2];return true;
}

static bool audio_mime(const char *s,size_t n) {
    return (n==9 && !strncasecmp(s,"audio/wav",9)) || (n==11 && !strncasecmp(s,"audio/x-wav",11)) ||
           (n==10 && !strncasecmp(s,"audio/wave",10));
}
static bool decode_audio(const jval *value,size_t cap,clef_audio_samples *out,char *err,size_t errlen) {
    if(value->type==J_ARRAY)return clef_audio_from_json(value,cap,out,err,errlen);
    const jval *source=value;
    if(value->type==J_OBJECT) {
        const jval *type=json_get(value,"content_type");source=json_get(value,"base64");
        if(value->n!=2 || !type || type->type!=J_STRING || !audio_mime(type->s,type->len) ||
           !source || source->type!=J_STRING || (source->len>=5 && !strncasecmp(source->s,"data:",5)))
            return error(err,errlen,"audio object requires WAV content_type and bare base64");
    } else if(value->type==J_STRING) {
        if(value->len<5 || strncasecmp(value->s,"data:",5))return error(err,errlen,"audio strings must be WAV data URLs");
        size_t end=5;while(end<value->len && value->s[end]!=';' && value->s[end]!=',')end++;
        if(!audio_mime(value->s+5,end-5))return error(err,errlen,"audio data URL must declare WAV");
    } else return error(err,errlen,"audio requires samples, a WAV data URL or a base64 object");
    uint8_t *bytes=NULL;size_t length=0;
    if(!clef_base64_decode(source->s,source->len,&bytes,&length,err,errlen))return false;
    bool ok=clef_audio_decode(bytes,length,cap,out,err,errlen);free(bytes);return ok;
}

bool clef_omni_encode_audio(const jval *audio,int tracks,const clef_encode_opts *opts,size_t *reserve,
                            clef_record *record,char *err,size_t errlen) {
    if(!audio || audio->type==J_NULL || !audio->n)return true;
    const size_t n=audio->n;
    if(audio->type!=J_ARRAY || !opts->audio.token_id || n>INT_MAX || tracks<0 || (size_t)tracks>n ||
       (opts->audio.max_items>0 && n>(size_t)opts->audio.max_items))return error(err,errlen,"too many or unsupported audio items");
    const size_t wrappers=3*n-(size_t)tracks;
    if(*reserve>=(size_t)opts->max_length || wrappers+n>(size_t)opts->max_length-*reserve)
        return error(err,errlen,"audio cannot fit the remaining context");
    clef_audio_samples *samples=calloc(n,sizeof(*samples));
    clef_audio_features *features=calloc(n,sizeof(*features));
    if(!samples || !features){free(samples);free(features);return error(err,errlen,"out of memory");}
    bool ok=false;size_t total=0,longest=0;
    const size_t context_samples=((size_t)opts->max_length-*reserve-wrappers)*1280+160*n;
    for(size_t i=0;i<n;i++) {
        // Even with soft limits removed, no accepted clip can exceed the sample
        // count implied by the remaining token budget. Check before decoding.
        if(total>=context_samples){error(err,errlen,"audio samples exceed context budget");goto done;}
        size_t cap=context_samples-total;
        if(opts->audio.max_samples) {
            if(total>=opts->audio.max_samples){error(err,errlen,"total audio sample limit exceeded");goto done;}
            if(cap>opts->audio.max_samples-total)cap=opts->audio.max_samples-total;
        }
        if(!decode_audio(audio->items[i],cap,samples+i,err,errlen))goto done;
        if(samples[i].count>SIZE_MAX-total){error(err,errlen,"audio sample count overflow");goto done;}
        total+=samples[i].count;if(samples[i].count>longest)longest=samples[i].count;
    }
    size_t tokens=0,padded=longest/160;
    for(size_t i=0;i<n;i++) {
        size_t frames=(samples[i].count+159)/160;if(frames>padded)frames=padded;
        size_t count=13*(frames/100)+(frames%100+7)/8;
        if(count>(size_t)opts->max_length || tokens>(size_t)opts->max_length-count) {
            error(err,errlen,"audio exceeds context");goto done;
        }
        tokens+=count;
    }
    if(tokens+wrappers>(size_t)opts->max_length-*reserve){error(err,errlen,"audio exceeds remaining context");goto done;}
    if(!clef_audio_mel_batch(samples,(int)n,features,err,errlen))goto done;
    record->audio=calloc(n,sizeof(*record->audio));
    if(!record->audio){error(err,errlen,"out of memory");goto done;}
    record->n_audio=(int)n;
    for(size_t i=0;i<n;i++) {
        record->audio[i].feat=features[i];features[i]=(clef_audio_features){0};
        record->audio[i].video_index=i<n-(size_t)tracks?-1:(int)(i-(n-(size_t)tracks));
        record->audio[i].token_rows=malloc((size_t)record->audio[i].feat.n_tokens*sizeof(int32_t));
        if(!record->audio[i].token_rows){error(err,errlen,"out of memory");goto done;}
    }
    record->n_audio_tokens=(int32_t)tokens;*reserve+=tokens+wrappers;ok=true;
done:
    for(size_t i=0;i<n;i++){clef_audio_samples_free(samples+i);clef_audio_features_free(features+i);}
    free(samples);free(features);return ok;
}

bool clef_omni_encode_record_audio(const jval *request,const clef_encode_opts *opts,size_t *reserve,
                                   clef_record *record,char *err,size_t errlen) {
    const jval *audio=json_get(request,"audio"),*videos=json_get(request,"videos");
    if(audio && audio->type!=J_ARRAY)audio=NULL;  // the encoder refused truthy non-lists; falsy ones are absent
    size_t tracks=0,standalone=audio?audio->n:0;
    if(videos && videos->type==J_ARRAY)for(size_t i=0;i<videos->n;i++)tracks+=json_get(videos->items[i],"audio")!=NULL;
    if(!tracks)return clef_omni_encode_audio(audio,0,opts,reserve,record,err,errlen);
    if(tracks!=videos->n)return error(err,errlen,"all videos must supply soundtracks or none may supply them");
    if(tracks>INT_MAX || standalone>INT_MAX-tracks ||
       (opts->audio.max_items>0 && standalone+tracks>(size_t)opts->audio.max_items))
        return error(err,errlen,"too many audio items including video soundtracks");
    jval joined={.type=J_ARRAY,.n=standalone+tracks};
    joined.items=calloc(joined.n,sizeof(*joined.items));
    if(!joined.items)return error(err,errlen,"out of memory");
    for(size_t i=0;i<standalone;i++)joined.items[i]=audio->items[i];
    for(size_t i=0;i<tracks;i++)joined.items[standalone+i]=json_get(videos->items[i],"audio");
    bool ok=clef_omni_encode_audio(&joined,(int)tracks,opts,reserve,record,err,errlen);
    free(joined.items);return ok;
}

bool clef_omni_positions(clef_record *record, char *err, size_t errlen) {
    if (!record->n_images) return true;
    if (record->pos3 || record->ids.len > INT32_MAX || record->ids.len > SIZE_MAX/(3*sizeof(int32_t)))
        return error(err, errlen, "invalid position allocation");
    int32_t *positions = malloc(record->ids.len*3*sizeof(*positions));
    if (!positions) return error(err, errlen, "out of memory for positions");
    size_t token = 0;
    int next = 0;
    int64_t current = 0;
    while (token < record->ids.len) {
        if (next < record->n_images && record->images[next].tok_start == (int32_t)token) {
            const int group = record->images[next].video_group;
            const clef_audio_ref *track=video_audio(record,group);
            int audio_index=0;
            int frame = 0;
            int64_t max_position = current;
            do {
                const clef_image_ref *image = &record->images[next];
                const int h = image->pt.grid_h/2, w = image->pt.grid_w/2;
                const int64_t time = (int64_t)frame*13;
                while(track && audio_index<track->feat.n_tokens && audio_index<time) {
                    if(token>=record->ids.len || track->token_rows[audio_index]!=(int32_t)token || current+audio_index>INT32_MAX)goto invalid;
                    positions[3*token]=positions[3*token+1]=positions[3*token+2]=(int32_t)(current+audio_index);
                    max_position=current+audio_index;audio_index++;token++;
                }
                if (image->tok_start != (int32_t)token || h <= 0 || w <= 0 || image->pt.grid_h%2 || image->pt.grid_w%2 ||
                    (int64_t)h*w != image->pt.n_tokens || (size_t)image->pt.n_tokens > record->ids.len-token ||
                    current+time > INT32_MAX || current+h > INT32_MAX || current+w > INT32_MAX) goto invalid;
                for (int y = 0; y < h; y++) for (int x = 0; x < w; x++, token++) {
                    positions[3*token] = (int32_t)(current+time);
                    positions[3*token+1] = (int32_t)(current+y);
                    positions[3*token+2] = (int32_t)(current+x);
                }
                // The reference continues after its last appended segment.
                // A trailing audio segment can reset the maximum even when an
                // earlier frame had a larger spatial coordinate.
                max_position=current;
                if (current+time > max_position) max_position = current+time;
                if (current+h-1 > max_position) max_position = current+h-1;
                if (current+w-1 > max_position) max_position = current+w-1;
                next++; frame++;
            } while (group && next < record->n_images && record->images[next].video_group == group);
            while(track && audio_index<track->feat.n_tokens) {
                if(token>=record->ids.len || track->token_rows[audio_index]!=(int32_t)token || current+audio_index>INT32_MAX)goto invalid;
                positions[3*token]=positions[3*token+1]=positions[3*token+2]=(int32_t)(current+audio_index);
                max_position=current+audio_index;audio_index++;token++;
            }
            current = max_position+1;
        } else {
            if (current > INT32_MAX || (next < record->n_images && record->images[next].tok_start < (int32_t)token)) goto invalid;
            positions[3*token] = positions[3*token+1] = positions[3*token+2] = (int32_t)current;
            current++; token++;
        }
    }
    if (next != record->n_images) goto invalid;
    record->pos3 = positions;
    return true;
invalid:
    free(positions);
    return error(err, errlen, "visual placement or rotary position overflow");
}
