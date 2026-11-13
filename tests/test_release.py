"""Tests for the release workflow's build and tag-validation gate."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
RELEASE_SCRIPT = ROOT / "scripts" / "release.py"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
TESTS_WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"


def load_release_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("monolingual_release", RELEASE_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


release = load_release_module()


def make_repo(tmp_path: Path, version: str = "1.2.3") -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    for name in release.PLUGIN_FILES:
        shutil.copy2(ROOT / name, repo / name)
    manifest = repo / "plugin.yaml"
    text = manifest.read_text(encoding="utf-8")
    text = release.PLUGIN_VERSION_PATTERN.sub(f"version: {version}", text, count=1)
    manifest.write_text(text, encoding="utf-8")
    return repo


def test_current_repository_builds_the_expected_archive(tmp_path: Path) -> None:
    version = release.read_plugin_version(ROOT)
    archive, digest = release.build_release(ROOT, f"v{version}", tmp_path / "dist")

    assert archive.name == f"monolingual-{version}.zip"
    assert release.sha256(archive) == digest
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == set(release.PLUGIN_FILES)
        assert "tests/test_release.py" not in bundle.namelist()
        assert "__pycache__" not in " ".join(bundle.namelist())
        assert f"version: {version}" in bundle.read("plugin.yaml").decode("utf-8")


def test_release_archive_is_deterministic(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    _, first = release.build_release(repo, "v1.2.3", tmp_path / "dist")
    _, second = release.build_release(repo, "v1.2.3", tmp_path / "dist")
    assert first == second


def test_tag_must_match_plugin_version(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, version="1.2.3")
    with pytest.raises(release.ReleaseError, match="does not match"):
        release.build_release(repo, "v1.2.4", tmp_path / "dist")
    assert not (tmp_path / "dist" / "monolingual-1.2.4.zip").exists()


@pytest.mark.parametrize("tag", ["1.2.3", "release-1.2.3", "v1.2", "v1.2.3.4", "v"])
def test_invalid_tag_is_rejected(tmp_path: Path, tag: str) -> None:
    repo = make_repo(tmp_path)
    with pytest.raises(release.ReleaseError):
        release.build_release(repo, tag, tmp_path / "dist")


def test_missing_plugin_file_blocks_release(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "translator.py").unlink()
    with pytest.raises(release.ReleaseError, match="translator.py"):
        release.build_release(repo, "v1.2.3", tmp_path / "dist")


def test_cli_exits_nonzero_for_mismatched_tag(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(RELEASE_SCRIPT),
            "--root",
            str(repo),
            "--tag",
            "v9.9.9",
            "--output",
            str(tmp_path / "dist"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "release blocked" in result.stderr
    assert not (tmp_path / "dist" / "monolingual-9.9.9.zip").exists()


def load_workflow(path: Path) -> dict:
    # BaseLoader keeps the YAML 1.1 ``on`` key as a string.
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_release_is_tag_triggered() -> None:
    workflow = load_workflow(RELEASE_WORKFLOW)
    assert "workflow_dispatch" not in workflow["on"]
    assert workflow["on"]["push"]["tags"] == ["v*"]


def test_release_cannot_publish_without_the_test_gate() -> None:
    workflow = load_workflow(RELEASE_WORKFLOW)
    release_job = workflow["jobs"]["release"]
    assert set(release_job["needs"]) == {"precheck", "tests"}
    assert workflow["jobs"]["tests"]["uses"] == "./.github/workflows/tests.yml"
    assert workflow["jobs"]["tests"]["needs"] == "precheck"
    assert release_job["permissions"]["contents"] == "write"


def test_release_precheck_validates_tag_before_running_tests() -> None:
    workflow = load_workflow(RELEASE_WORKFLOW)
    precheck = workflow["jobs"]["precheck"]
    commands = "\n".join(step.get("run", "") for step in precheck["steps"])
    assert 'scripts/release.py --tag "$GITHUB_REF_NAME"' in commands


def test_reusable_tests_cover_lingua_and_fallback_modes() -> None:
    workflow = load_workflow(TESTS_WORKFLOW)
    assert "workflow_call" in workflow["on"]
    matrix = workflow["jobs"]["test"]["strategy"]["matrix"]["include"]
    assert {entry["python"] for entry in matrix} >= {"3.11", "3.12", "3.13", "3.14"}
    assert {entry["lingua"] for entry in matrix} == {"true", "false"}
