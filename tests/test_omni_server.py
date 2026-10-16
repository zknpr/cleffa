"""Real Omni HTTP validation, recovery, cache fallback and co-batch retry."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import unittest

from test_omni_runtime import REQUEST, ROOT

MODEL = ROOT/'gguf/clef-omni-text.gguf'
BASE_ENV = {k:v for k,v in os.environ.items() if not k.startswith('CLEF_')}


class OmniServer(unittest.TestCase):
    model=MODEL
    @contextmanager
    def server(self, *args, env=None):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            port = sock.getsockname()[1]
        with tempfile.TemporaryFile(mode='w+') as log:
            process = subprocess.Popen([ROOT/'clef-server','-m',self.model,'--port',str(port),
                                        '--no-warmup','--no-keep-warm',*args],
                                       stdout=subprocess.DEVNULL, stderr=log, env={**BASE_ENV,**(env or {})})
            try:
                deadline = time.monotonic()+60
                while True:
                    if process.poll() is not None:
                        log.seek(0)
                        self.fail('server exited: '+log.read())
                    try:
                        with socket.create_connection(('127.0.0.1',port), timeout=.5): break
                    except OSError:
                        if time.monotonic()>deadline: self.fail('server startup timeout')
                        time.sleep(.05)
                yield port, log
            finally:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()

    def post(self, port, req, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1',port,timeout=300)
        try:
            connection.request('POST','/v1/systemone',json.dumps(req),headers=headers or {})
            response=connection.getresponse()
            return response.status,json.loads(response.read())
        finally: connection.close()

    def expected(self, req):
        p=subprocess.run([ROOT/'clef','-m',self.model,'--strict'],input=json.dumps(req)+'\n',
                         text=True,capture_output=True,check=True,env=BASE_ENV)
        return json.loads(p.stdout)

    def test_recovery_validation_and_cache_fallback(self):
        expected=self.expected(REQUEST)
        with self.server('--prefix-cache-mb','64',env={'CLEF_DEBUG_NIL_CMDBUF':'1'}) as (port,log):
            for media in ('audio','images','videos'):
                status,body=self.post(port,{**REQUEST,media:[{}]})
                self.assertEqual(status,400)
                self.assertIn('error',body)
            status,body=self.post(port,{**REQUEST,'state':' alpha'*65000})
            self.assertEqual(status,400)
            self.assertIn('64000',json.dumps(body))
            status,body=self.post(port,REQUEST)
            self.assertEqual(status,500)
            self.assertIn('command buffer',json.dumps(body))
            for headers in ({},{'X-Clef-Prefix-Cache':'tenant-a'},{'X-Clef-Prefix-Cache':'tenant-a'}):
                status,body=self.post(port,REQUEST,headers)
                self.assertEqual((status,body),(200,expected))
            log.seek(0)
            text=log.read()
            self.assertIn('reused 0 of',text)

    def test_co_batched_requests_retry_independently(self):
        expected=self.expected(REQUEST)
        with self.server('--batch','8',env={'CLEF_DEBUG_FAIL_MULTI':'1'}) as (port,log):
            barrier=threading.Barrier(9)
            def send():
                barrier.wait(timeout=30)
                return self.post(port,REQUEST)
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures=[pool.submit(send) for _ in range(8)]
                barrier.wait(timeout=30)
                result=[f.result() for f in futures]
            self.assertEqual(result,[(200,expected)]*8)
            log.seek(0)
            self.assertIn('injected batch failure',log.read())

    def test_explicit_context_limit(self):
        with self.server('--max-length','300') as (port,log):
            status,body=self.post(port,{**REQUEST,'state':' alpha'*300})
            self.assertEqual(status,400)
            self.assertIn('max 300 tokens',json.dumps(body))


if __name__ == '__main__': unittest.main()
