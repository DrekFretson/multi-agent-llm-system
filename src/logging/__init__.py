"""Project JSONL logging utilities.

The project package is imported as ``src.logging``. If a tool puts the
``src`` directory itself on ``sys.path`` and resolves this package as the
top-level ``logging`` module, delegate to Python's standard-library package
instead of shadowing its API.
"""

if __name__ == "logging":
    from pathlib import Path
    import sysconfig

    _stdlib_logging_path = Path(sysconfig.get_path("stdlib")) / "logging"
    __path__.append(str(_stdlib_logging_path))
    _stdlib_logging_init = _stdlib_logging_path / "__init__.py"
    __file__ = str(_stdlib_logging_init)
    exec(compile(_stdlib_logging_init.read_bytes(), __file__, "exec"), globals())
else:
    from .logger import Logger, LoggerError, LoggerSerializationError, LoggerWriteError

    __all__ = [
        "Logger",
        "LoggerError",
        "LoggerSerializationError",
        "LoggerWriteError",
    ]
