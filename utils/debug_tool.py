import signal,pystack,sys,traceback,threading
# pystack
def pystack():
    for tid, stack in sys._current_frames().items():
        info = []
        t = _get_thread(tid)
        info.append('"%s" tid=%d' % (t.name, tid))
        for filename, lineno, _, line in traceback.extract_stack(stack):
            info.append('    at %s(%s:%d)' % (line, filename[filename.rfind('/') + 1:], lineno))
        print ('\r\n'.join(info))
        print ('')

def _get_thread(tid):
    for t in threading.enumerate():
        if t.ident == tid:
            return t
    return None

def _pystack(sig, frame):
    pystack()

def enable_pystack():
    signal.signal(signal.SIGUSR1, _pystack)