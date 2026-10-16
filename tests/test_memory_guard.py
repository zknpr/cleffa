"""The watchdog must abort/reap an oversized child and leave failed evidence."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import psutil

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from memory_guard import footprint,run,violation


class MemoryGuard(unittest.TestCase):
    def test_limits_and_measurement(self):
        self.assertIsNone(violation(100,200,100,200))
        self.assertIn('budget',violation(101,200,100,200))
        self.assertIn('headroom',violation(1,199,100,200))
        import os
        self.assertGreater(footprint(os.getpid()),0)

    def test_success_and_memory_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            result=run([sys.executable,'-c','print("small child")'],64<<20,1,folder/'ok.json')
            self.assertTrue(result['complete'])
            # 64 MiB is intentionally small; no model or GPU allocation occurs.
            result=run([sys.executable,'-c','import time; a=bytearray(64*1024*1024); time.sleep(10)'],
                       32<<20,1,folder/'failed.json')
            self.assertFalse(result['complete'])
            self.assertIn('budget',result['error'])
            self.assertLess(result['seconds'],5)
            self.assertFalse(json.loads((folder/'failed.json').read_text())['complete'])

    def test_termination_reaps_child_and_grandchild(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);pids=folder/'pids.json';report=folder/'report.json'
            child=('import json,os,subprocess,sys,time; '
                   'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"],start_new_session=True); '
                   'open(sys.argv[1],"w").write(json.dumps([os.getpid(),p.pid])); time.sleep(30)')
            guard=subprocess.Popen([sys.executable,str(Path(__file__).resolve().parents[1]/'tools/memory_guard.py'),
                '--max-gib','1','--reserve-gib','.001','--report',str(report),'--',
                sys.executable,'-c',child,str(pids)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            owned=[]
            try:
                deadline=time.monotonic()+5
                while not pids.exists() and time.monotonic()<deadline:time.sleep(.02)
                self.assertTrue(pids.exists(),'guarded child did not start')
                owned=json.loads(pids.read_text())
                guard.send_signal(signal.SIGTERM)
                guard.wait(timeout=5)
                self.assertNotEqual(guard.returncode,0)
                result=json.loads(report.read_text())
                self.assertFalse(result['complete'])
                self.assertIn('SIGTERM',result.get('error',''))
                deadline=time.monotonic()+2
                def running(pid):
                    try:return psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE
                    except psutil.NoSuchProcess:return False
                while any(running(pid) for pid in owned) and time.monotonic()<deadline:time.sleep(.02)
                self.assertFalse(any(running(pid) for pid in owned),'detached workload survived guard termination')
            finally:
                # Keep the regression safe even against the broken implementation.
                for pid in owned:
                    try:os.kill(pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                if guard.poll() is None:guard.kill()
                guard.communicate(timeout=5)

    def test_tracked_orphan_still_counts_toward_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            orphan='import time; time.sleep(.9); a=bytearray(128*1024*1024); time.sleep(10)'
            middle=f'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{orphan!r}],start_new_session=True); time.sleep(.4)'
            parent=f'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{middle!r}]); time.sleep(10)'
            result=run([sys.executable,'-c',parent],96<<20,1,Path(tmp)/'orphan.json',timeout=3)
            self.assertFalse(result['complete'])
            self.assertIn('budget',result['error'])


if __name__=='__main__':unittest.main()
