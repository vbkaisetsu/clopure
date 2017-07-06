import shlex
import sys
import traceback
import socket

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


def clopure_listen_iter_connection(host, port, backlog):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, port))
    sock.listen(backlog)
    sock.settimeout(0.5)
    while True:
        try:
            conn, addr = sock.accept()
            yield conn
        except socket.timeout:
            pass


functions = {
    "system-map": clopure_system_map,
    "listen-iter-connection": clopure_listen_iter_connection,
}
