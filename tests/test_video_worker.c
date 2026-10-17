/* Process failures must be bounded, observable, and leave no child behind. */
#define OUTPUT_CAP 64
#define DECODE_SECONDS .15
#include "../clef_encoded_video.c"
#include <assert.h>
int main(void) {
    char err[512];jbuf out={0};
    char *good[]={"/bin/cat",NULL};
    assert(worker((const uint8_t *)"abc",3,good,&out,err,sizeof(err)));
    assert(out.len==3&&!memcmp(out.p,"abc",3));jbuf_free(&out);
    char *large[]={"/usr/bin/yes",NULL};
    assert(!worker(NULL,0,large,&out,err,sizeof(err)));
    assert(strstr(err,"output exceeds"));jbuf_free(&out);
    char *slow[]={"/bin/sleep","2",NULL};
    assert(!worker(NULL,0,slow,&out,err,sizeof(err)));
    assert(strstr(err,"deadline"));jbuf_free(&out);
    char *failed[]={"/bin/sh","-c","printf 'bad\\ninput' >&2; exit 7",NULL};
    assert(!worker(NULL,0,failed,&out,err,sizeof(err)));
    assert(strstr(err,"bad input")&&!strchr(err,'\n'));jbuf_free(&out);
    int status;assert(waitpid(-1,&status,WNOHANG)==-1&&errno==ECHILD);
    /* The worker parses attacker-controlled files: the server's environment, which can hold
     * deployment secrets, must not reach it. env prints nothing for an empty environment. */
    assert(!setenv("CLEF_TEST_SECRET","leak",1));
    char *env[]={"/usr/bin/env",NULL};
    assert(worker(NULL,0,env,&out,err,sizeof(err)));
    assert(out.len==0);jbuf_free(&out);
    char *missing[]={"/does-not-exist/clef-media",NULL};
    assert(!worker(NULL,0,missing,&out,err,sizeof(err)));
    assert(strstr(err,"make media"));jbuf_free(&out);
    puts("worker output, deadline, error and reap checks passed");return 0;
}
