"""Manifest declaration guarantees (audit M5 / decision 4).

The Hermes runtime parser does not model ``provides_middleware`` (it warns about an
unknown field), but ``hermes plugins validate`` REJECTS middleware registered without the
declaration. The committed manifest therefore declares it and the mismatch is proven here.
"""

from pathlib import Path

import pytest

from jev_fastpath.config import SettingsError, load_settings

REPO_ROOT = Path(__file__).resolve().parents[1]


def _raw_manifest():
    import yaml

    return yaml.safe_load((REPO_ROOT / "plugin.yaml").read_text(encoding="utf-8"))


class TestManifestDeclarations:
    def test_middleware_is_declared(self):
        manifest = _raw_manifest()
        assert manifest["provides_middleware"] == ["llm_execution"]

    def test_requires_hermes_is_declared(self):
        manifest = _raw_manifest()
        assert manifest["requires_hermes"] == ">=0.21.3"

    def test_single_registered_surface(self):
        manifest = _raw_manifest()
        assert manifest["provides_tools"] == []
        assert manifest["provides_hooks"] == []

    def test_hermes_parser_tolerates_the_declaration(self):
        """The upstream parser warns on the unknown field but parses successfully."""
        pytest.importorskip(
            "hermes_cli.plugins",
            reason="Hermes runtime required; run with PYTHONPATH pointing at the Hermes checkout",
        )
        from hermes_cli.plugins import parse_manifest_file

        manifest = parse_manifest_file(REPO_ROOT / "plugin.yaml", REPO_ROOT, "test", "")
        assert manifest is not None
        assert manifest.name == "jev-fastpath"
        assert manifest.manifest_version == 2
        assert manifest.kind == "standalone"

    def test_hermes_validator_passes_on_this_repo(self):
        pytest.importorskip(
            "hermes_cli.plugin_validate",
            reason="Hermes runtime required; run with PYTHONPATH pointing at the Hermes checkout",
        )
        from hermes_cli.plugin_validate import validate_plugin_dir

        report = validate_plugin_dir(REPO_ROOT)
        failed = [name for name, ok, _detail in report.checks if not ok]
        assert report.ok, failed


class TestRootEntrypoint:
    """L4: one lazy, relative resolution path — no absolute fallback to hide behind."""

    def test_no_absolute_import_fallback(self):
        source = (REPO_ROOT / "__init__.py").read_text(encoding="utf-8")
        assert "from .jev_fastpath.plugin import register" in source
        assert "from jev_fastpath.plugin import register" not in source

    def test_standalone_import_executes_nothing_hostile(self):
        # Importing the file without package context (as pytest 9 package nodes do) must
        # succeed lazily: no relative import runs at module-execution time.
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "jev_fastpath_standalone_root", REPO_ROOT / "__init__.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            assert module.__all__ == ["register"]
            # Lazy: no import side effects executed during exec_module.
            assert "register" not in module.__dict__
        finally:
            sys.modules.pop(spec.name, None)

    def test_register_resolves_through_package_context(self):
        """When imported as a package (as Hermes does), register resolves."""
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "jev_fastpath_root_probe", REPO_ROOT / "__init__.py",
            submodule_search_locations=[str(REPO_ROOT)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            assert callable(module.register)  # relative import resolves via __path__
        finally:
            sys.modules.pop(spec.name, None)
