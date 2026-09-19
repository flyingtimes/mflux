import pytest

from mflux.utils.version_util import VersionUtil


def _module_file(root, *parts):
    # A stand-in for version_util.py somewhere under `root`; the scan walks up from its parent.
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


@pytest.mark.fast
def test_scan_reads_mflux_own_pyproject_in_a_source_checkout(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "mflux"\nversion = "1.2.3"\n')
    module = _module_file(tmp_path, "src", "mflux", "utils", "version_util.py")

    assert VersionUtil._scan_pyproject(module) == "1.2.3"


@pytest.mark.fast
def test_scan_skips_a_host_project_pyproject_above_a_vendored_copy(tmp_path):
    # mflux installed as a dependency: the first pyproject above site-packages is the host's (#728).
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "host-app"\nversion = "0.1.0"\n')
    module = _module_file(tmp_path, ".venv", "lib", "python3.12", "site-packages", "mflux", "utils", "version_util.py")

    assert VersionUtil._scan_pyproject(module) is None


@pytest.mark.fast
def test_scan_returns_none_without_any_pyproject(tmp_path):
    module = _module_file(tmp_path, "site-packages", "mflux", "utils", "version_util.py")

    assert VersionUtil._scan_pyproject(module) is None


@pytest.mark.fast
def test_version_falls_back_to_installed_metadata_when_the_scan_finds_a_host(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "host-app"\nversion = "0.1.0"\n')
    module = _module_file(tmp_path, ".venv", "lib", "python3.12", "site-packages", "mflux", "utils", "version_util.py")
    scan = VersionUtil._scan_pyproject  # the real scan, started from the vendored copy's location
    monkeypatch.setattr(VersionUtil, "_scan_pyproject", staticmethod(lambda: scan(module)))
    monkeypatch.setattr(VersionUtil, "_get_installed_version", staticmethod(lambda: "9.8.7"))

    assert VersionUtil.get_mflux_version() == "9.8.7"


@pytest.mark.fast
def test_version_reports_unknown_when_nothing_answers(monkeypatch):
    monkeypatch.setattr(VersionUtil, "_scan_pyproject", staticmethod(lambda: None))
    monkeypatch.setattr(VersionUtil, "_get_installed_version", staticmethod(lambda: None))

    assert VersionUtil.get_mflux_version() == "unknown"
