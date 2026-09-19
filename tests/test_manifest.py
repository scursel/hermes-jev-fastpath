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
