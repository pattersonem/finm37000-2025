import contextlib
import pathlib
import os
import re
from typing import Generator


@contextlib.contextmanager
def temp_env(**environ: str) -> Generator:
    """
    Temporarily set the process environment variables.

    >>> with temp_env(PLUGINS_DIR='test/plugins'):
    ...   "PLUGINS_DIR" in os.environ
    True

    >>> "PLUGINS_DIR" in os.environ
    False

    :type environ: dict[str, unicode]
    :param environ: Environment variables to set
    """
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


class Secret(str):
    def __str__(self) -> str:
        return re.sub(r"\S", "*", self)


def get_databento_api_key(
    path: pathlib.Path = pathlib.Path.home() / ".databento_api_key",
) -> Secret:
    """
    Retrieve the databento API key from a file.

    :param path: Path to the file with the API key. Default is `~/.databento_api_key`.

    Reads and return one line of the text file specified by `path`.
    Save your api key in the default `~/.databento_api_key` file
    and make that file only accessible to your user. Then you can
    create a databento historical client without leaving your API key
    in your code. For example,

    >>> import databento as db  # doctest: +SKIP
    >>> client = db.Historical(get_databento_api_key())  # doctest: +SKIP

    or

    >>> with temp_env(DATABENTO_API_KEY=get_databento_api_key()):
    ...     client = db.Historical()  # doctest: +SKIP

    """
    with open(path) as f:
        api_key = f.readline().strip()
    return Secret(api_key)


if __name__ == "__main__":
    with temp_env(DATABENTO_API_KEY=get_databento_api_key()):
        secret = Secret(os.environ.get("DATABENTO_API_KEY"))
    print(f"Your secret was read: {secret}")
    print(f"Your secret is not in the env: {os.environ.get('DATABENTO_API_KEY')}")
