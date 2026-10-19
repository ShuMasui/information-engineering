import subprocess

from lights_out.infrastructure import provenance


def git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_collect_provenance_has_the_expected_keys():
    info = provenance.collect_provenance()
    assert set(info) == {"package_version", "python_version", "platform", "git"}
    assert info["package_version"] == "0.1.0"


def test_git_info_reports_commit_and_dirty_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "_PACKAGE_DIR", tmp_path)
    git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("a")
    git(tmp_path, "add", "a.txt")
    git(tmp_path, "commit", "-q", "-m", "first")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout.strip()

    assert provenance.collect_provenance()["git"] == {"commit": head, "dirty": False}

    (tmp_path / "a.txt").write_text("changed")
    assert provenance.collect_provenance()["git"] == {"commit": head, "dirty": True}


def test_git_info_is_none_outside_a_repository(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "_PACKAGE_DIR", tmp_path)
    assert provenance.collect_provenance()["git"] is None
