import os
import sys

from .cli import main

code = main()
for stream in (sys.stdout, sys.stderr):
    try:
        stream.flush()
    except (OSError, ValueError):
        pass
# The answer is on disk; a model call still in flight must not hold the container open until it times out.
os._exit(code)
