#!/usr/bin/env python3

import socket
import sys


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("localhost", 1234))
sock.send(("%s\n" % sys.argv[1]).encode("utf-8"))
print("received:", sock.recv(1024).decode("utf-8"))
