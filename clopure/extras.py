import shlex
import sys
import traceback

from subprocess import Popen, PIPE, STDOUT
from clopure.exceptions import ClopureRuntimeError


def clopure_system_map(command, g):
    proc = Popen(shlex.split(command), stdout=PIPE, stdin=PIPE, stderr=None)
    try:
        for data in g:
            if not isinstance(data, bytes):
                proc.kill()
                raise ClopureRuntimeError("bytes is expected")
            proc.stdin.write(data)
            proc.stdin.flush()
            result = proc.stdout.readline()
            yield result
    except BrokenPipeError:
        traceback.print_exc(file=sys.stdout)
    proc.kill()


functions = {
    "system-map": clopure_system_map,
}
