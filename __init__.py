"""Hermes directory-plugin entrypoint: the plugin exposes exactly one ``register(ctx)``."""

try:
    from .jev_fastpath.plugin import register
except ImportError:
    # Hermes loads this directory as a package, so the relative import is the real path.
    # Tooling that imports the file standalone (e.g. pytest 9 package nodes) has no parent
    # package; there the absolute import resolves from the repository root on sys.path.
    from jev_fastpath.plugin import register

__all__ = ["register"]
