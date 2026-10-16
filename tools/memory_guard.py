"""Run one process group with a sampled memory-abort threshold and OS headroom.

The watchdog complements an allocator limit; it cannot prevent a single oversized
allocation between samples. On macOS use physical footprint, which includes Metal
allocations that RSS alone can miss. Child audit probes share the same budget.
"""
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


class MacUsage(ctypes.Structure):
    # RUSAGE_INFO_V0 from the macOS SDK's sys/resource.h.
    _fields_=[('uuid',ctypes.c_ubyte*16)]+[(name,ctypes.c_uint64) for name in
        ('user','system','wakeups','interrupts','pageins','wired','resident','footprint','start','exit')]


def footprint(pid):
    if sys.platform!='darwin':return psutil.Process(pid).memory_info().rss
    library=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
    library.proc_pid_rusage.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
    library.proc_pid_rusage.restype=ctypes.c_int
    usage=MacUsage()
    if library.proc_pid_rusage(pid,0,ctypes.byref(usage)):
        code=ctypes.get_errno()
        if not psutil.pid_exists(pid):raise psutil.NoSuchProcess(pid)
        raise OSError(code,'cannot measure process physical footprint')
    return max(usage.footprint,usage.resident)


def violation(used,available,limit,reserve):
    if used>limit:return 'process group exceeded memory budget'
    if available<reserve:return 'system memory headroom fell below reserve'
    return None


def run(command,limit,reserve,report_path,*,timeout=7200):
    report=dict(complete=False,command=command,limit_bytes=limit,reserve_bytes=reserve,
                measurement='sum of max(physical footprint, RSS) for process and descendants',
                sample_seconds=.1,peak_bytes=0,returncode=None,
                min_available_bytes=psutil.virtual_memory().available)
    report_path=Path(report_path)
    if report_path.exists():raise ValueError('refusing to replace memory evidence')
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    process=None;parent=None;owned={};start=time.monotonic();interrupted=None
    def on_signal(signum,frame):
        nonlocal interrupted
        # Do not raise inside Popen: a signal between fork and assignment could
        # otherwise leave an untracked child. The monitoring loop owns cleanup.
        interrupted=signum
    handlers={sig:signal.signal(sig,on_signal) for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT)}
    try:
        if psutil.virtual_memory().available<reserve:raise MemoryError('insufficient system memory headroom before launch')
        process=subprocess.Popen(command,start_new_session=True)
        parent=psutil.Process(process.pid)
        while process.poll() is None:
            if interrupted is not None:raise InterruptedError(f'interrupted by {signal.Signals(interrupted).name}')
            used=0
            try:members=[parent,*parent.children(recursive=True)]
            except psutil.NoSuchProcess:members=[]
            for member in members:
                owned[member.pid]=member
            # A tracked helper remains part of the budget after its immediate
            # parent exits. is_running checks the cached creation-time identity.
            for member in list(owned.values()):
                if not member.is_running():
                    owned.pop(member.pid,None)
                    continue
                try:used+=footprint(member.pid)
                except psutil.NoSuchProcess:pass
            report['peak_bytes']=max(report['peak_bytes'],used)
            available=psutil.virtual_memory().available
            report['min_available_bytes']=min(report['min_available_bytes'],available)
            report['last_available_bytes']=available
            report['last_usage_bytes']=used
            reason=violation(used,available,limit,reserve)
            if reason or time.monotonic()-start>timeout:
                raise MemoryError(reason or 'guarded command exceeded time limit')
            time.sleep(.1)
        if interrupted is not None:raise InterruptedError(f'interrupted by {signal.Signals(interrupted).name}')
        report['returncode']=process.wait()
        report['complete']=report['returncode']==0
    except (OSError,MemoryError,psutil.Error) as exc:
        report['error']=str(exc)
        print(f'memory guard: {exc}; peak {report["peak_bytes"]/(1<<30):.2f} GiB',file=sys.stderr,flush=True)
    finally:
        if process is not None:
            # Helpers can start new sessions, outside the main process group.
            # Retain Process identities across samples so reparenting does not
            # lose them; psutil's kill checks protect against PID reuse.
            try:
                if parent is not None:
                    for child in parent.children(recursive=True):owned[child.pid]=child
            except psutil.NoSuchProcess:pass
            for child in reversed(list(owned.values())):
                if child.pid==process.pid:continue
                try:child.kill()
                except psutil.NoSuchProcess:pass
            # This group was created exclusively for this invocation. Also reap
            # audit descendants if their parent failed or was interrupted.
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
        report['seconds']=time.monotonic()-start
        report_path.write_text(json.dumps(report,indent=2)+'\n')
        for sig,handler in handlers.items():signal.signal(sig,handler)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-gib',type=float,required=True)
    parser.add_argument('--reserve-gib',type=float,default=32.)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=7200.)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if not all(math.isfinite(v) and v>0 for v in (args.max_gib,args.reserve_gib,args.timeout)):
        parser.error('memory budgets and timeout must be finite and positive')
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:parser.error('missing guarded command')
    limit,reserve=(int(v*(1<<30)) for v in (args.max_gib,args.reserve_gib))
    if limit+reserve>psutil.virtual_memory().total:parser.error('budget plus reserve exceeds physical RAM')
    result=run(command,limit,reserve,args.report,timeout=args.timeout)
    raise SystemExit(0 if result['complete'] else 86)


if __name__=='__main__':main()
