import os
from contextlib import contextmanager

@contextmanager
def temp_env(**kwargs):
    sentinel = object()
    old = {k: os.environ.get(k, sentinel) for k in kwargs}
    try:
        for k, v in kwargs.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is sentinel:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

def get_databento_api_key() -> str:
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("DATABENTO_API_KEY not set")
    return key
