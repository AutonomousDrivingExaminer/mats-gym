import psutil


def is_used(port):
    """Checks whether a port is used"""
    return port in [conn.laddr.port for conn in psutil.net_connections()]


def get_next_free_port(port):
    return next(filter(lambda p: not is_used(p), range(port, 32000)))
