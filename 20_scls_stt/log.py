import time
import sys 
import inspect

def debug(*message):
    callerframerecord = inspect.stack()[1]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    tm = time.strftime("%Y-%m-%d %H:%M:%S")
    print('[DBG, %s %s %s:%s]' % (tm, info.filename.split('/')[-1], info.function, info.lineno), *message)
    sys.stdout.flush()

def info(*message):
    callerframerecord = inspect.stack()[1]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    tm = time.strftime("%Y-%m-%d %H:%M:%S")
    print('[INF, %s %s %s:%s]' % (tm, info.filename.split('/')[-1], info.function, info.lineno), *message)
    sys.stdout.flush()

def error(*message):
    callerframerecord = inspect.stack()[1]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    tm = time.strftime("%Y-%m-%d %H:%M:%S")
    print('[ERR, %s %s %s:%s]' % (tm, info.filename.split('/')[-1], info.function, info.lineno), *message)
    sys.stdout.flush()
