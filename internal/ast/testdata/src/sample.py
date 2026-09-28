import os
import subprocess

TIMEOUT = 30
CONFIG = {"debug": False}


def run(cmd):
    """Run a shell command."""
    return os.system(cmd)


@cache
def load(path):
    with open(path) as f:
        return f.read()


class Store:
    def __init__(self, root):
        self.root = root

    def get(self, key):
        def inner():
            return key
        return inner()


class Empty:
    pass
