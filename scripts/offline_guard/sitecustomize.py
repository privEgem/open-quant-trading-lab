"""Test-process tripwire, loaded through PYTHONPATH by verify_offline.py.

No firewall claim: this blocks Python socket calls, including child interpreters.
Mock transports and local file/SQLite work remain available.
"""
import socket


def denied(*args, **kwargs):
    raise RuntimeError("Real networking forbidden during offline release validation")


socket.socket.connect = denied
socket.socket.connect_ex = denied
socket.create_connection = denied
socket.getaddrinfo = denied
socket.socket.sendto = denied
