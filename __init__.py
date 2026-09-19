"""Hermes directory-plugin entrypoint: the plugin exposes exactly one ``register(ctx)``.

The import is deliberately lazy (PEP 562) with a single, relative resolution path — no
absolute fallback that could silently load a same-named package from elsewhere (audit L4).
Hermes loads this directory as a package and accesses ``register``; tooling that imports
the file standalone (pytest package nodes) imports the module without executing the
relative import.
"""

__all__ = ["register"]


def __getattr__(name: str):
    if name == "register":
        from .jev_fastpath.plugin import register

        return register
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
