#ifndef CLEF_ENCODED_VIDEO_H
#define CLEF_ENCODED_VIDEO_H
#include "clef_record.h"
/* Shallow request view: question strings retain their original owner. Only
 * media fields belong to arena and are consumed during request encoding. */
typedef struct { jval request; jmember *members; jarena *arena; jbuf storage; } clef_video_request;
bool clef_video_expand(const jval *request,clef_encode_opts opts,clef_video_request *out,char *err,size_t errlen);
void clef_video_request_free(clef_video_request *request);
#endif
