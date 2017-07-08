import socket


def listen_iter_connection(host, port, backlog):
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
