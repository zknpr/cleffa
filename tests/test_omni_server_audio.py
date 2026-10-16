"""Audio/AV HTTP admission, strict tokens, packing, disconnects and recovery."""
import copy
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
from pathlib import Path
import socket
import subprocess
import sys
import unittest

import test_omni_server as common

ROOT=common.ROOT
MODEL=Path(sys.argv.pop(1)) if len(sys.argv)>1 else ROOT/'gguf/clef-omni.gguf'
AUDIO=Path(sys.argv.pop(1)) if len(sys.argv)>1 else ROOT/'golden/clef-omni-audio-f32'
AV=Path(sys.argv.pop(1)) if len(sys.argv)>1 else ROOT/'golden/clef-omni-av-f32'


class AudioServer(unittest.TestCase):
    model=MODEL
    server=common.OmniServer.server
    post=common.OmniServer.post

    @classmethod
    def setUpClass(cls):
        def requests(path):return list(map(json.loads,(path/'requests.jsonl').read_text().splitlines()))
        cls.audio=requests(AUDIO)
        cls.av=requests(AV)

    def native(self,requests,*flags,env=None):
        p=subprocess.run([ROOT/'clef','-m',self.model,'--strict','--no-truncate',*flags],
                         input=''.join(json.dumps(r)+'\n' for r in requests),text=True,capture_output=True,
                         check=True,timeout=600,env={**common.BASE_ENV,**(env or {})})
        rows=list(map(json.loads,p.stdout.splitlines()))
        self.assertEqual(len(rows),len(requests));self.assertTrue(all('error' not in r for r in rows),rows)
        return rows

    def test_packing_poison_and_cache_fallback(self):
        requests=self.audio+self.av+[common.REQUEST,self.audio[0]]
        expected=self.native(requests,'--logits')
        for flags in (('--batch','8'),('--prefix-cache',),('--template-cache',)):
            self.assertEqual(self.native(requests,'--logits',*flags,env={'CLEF_DEBUG_POISON':'1'}),expected)

    def test_admission_limits_disconnect_and_strict_tokens(self):
        audio,video=self.audio[0],self.av[0]
        good=self.native([audio,video])
        bad=[dict(audio,audio=audio['audio']*5),dict(audio,audio=[[True]]),dict(audio,audio=['https://example.com/a.wav'])]
        mixed=copy.deepcopy(video);mixed['videos'].append(copy.deepcopy(mixed['videos'][0]));del mixed['videos'][1]['audio']
        bad.append(mixed)
        literal=dict(audio,state='<|audio_pad|><|audio_start|><|audio_end|>')
        expected_literal=self.native([literal])[0]
        with self.server('--max-image-requests','1','--max-audio-samples','80000') as (port,log):
            for request in bad:
                status,body=self.post(port,request);self.assertEqual(status,400,body)
                self.assertEqual(self.post(port,audio),(200,good[0]))
            # A disconnected client must release the same media admission slot.
            payload=json.dumps(video).encode()
            with socket.create_connection(('127.0.0.1',port),timeout=5) as sock:
                sock.sendall(b'POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nContent-Length: '+str(len(payload)).encode()+b'\r\n\r\n'+payload)
            self.assertEqual(self.post(port,literal),(200,expected_literal))
            with ThreadPoolExecutor(max_workers=4) as pool:
                results=list(pool.map(lambda i:self.post(port,[audio,video][i%2]),range(8)))
            self.assertEqual(results,[(200,good[i%2]) for i in range(8)])
        with self.server('--max-audio','1','--max-audio-samples','100') as (port,log):
            for request in (audio,video):self.assertEqual(self.post(port,request)[0],400)

    def test_failed_audio_command_recovers_and_keyed_requests_match(self):
        requests=[self.audio[0],self.av[0],common.REQUEST,self.audio[1]]
        expected=self.native(requests)
        with self.server('--max-image-requests','1','--prefix-cache-mb','64',
                         env={'CLEF_DEBUG_NIL_CMDBUF':'2','CLEF_DEBUG_POISON':'1'}) as (port,log):
            status,body=self.post(port,requests[0])
            self.assertEqual(status,500,body);self.assertIn('command buffer',json.dumps(body))
            for request,want in zip(requests,expected):
                self.assertEqual(self.post(port,request,{'X-Clef-Prefix-Cache':'audio'}),(200,want))

    def test_full_sample_budget_requires_sufficient_body_limit(self):
        import numpy as np
        sys.path.insert(0,str(ROOT/'ref'))
        from corpus_audio import wav
        request=dict(self.audio[0],audio=[wav(np.zeros(4800000,dtype='f4'),pcm16=True)])
        payload=json.dumps(request).encode()
        self.assertGreater(len(payload),8<<20)
        with self.server() as (port,log):
            # Exercise header rejection without transmitting a body after the
            # server has already closed the oversized request connection.
            connection=http.client.HTTPConnection('127.0.0.1',port,timeout=30)
            try:
                connection.putrequest('POST','/v1/systemone')
                connection.putheader('Content-Length',str(len(payload)));connection.endheaders()
                response=connection.getresponse();self.assertEqual(response.status,413);response.read()
            finally:connection.close()
        expected=self.native([request])[0]
        with self.server('--max-body',str(16<<20),'--max-image-requests','1') as (port,log):
            self.assertEqual(self.post(port,request),(200,expected))
            too_long=dict(request,audio=[wav(np.zeros(4800001,dtype='f4'),pcm16=True)])
            status,body=self.post(port,too_long);self.assertEqual(status,400,body)
            self.assertEqual(self.post(port,self.audio[0])[0],200)


if __name__=='__main__':unittest.main()
