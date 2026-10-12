"""Consistency checks before publishing torchsx and torchsx-native.

usage: python .github/scripts/check_release.py [TAG]

- TAG (a GitHub release tag) must be ``v<torchsx version>``, and
  ``torchsx.__version__`` must be that version.
- torchsx's ``torchsx-native`` requirement must accept native/Cargo.toml's version.
- ``PROTOCOL`` in native/src/lib.rs and torchsx/io/_native.py must agree.
- torchsx-native is uploaded with skip-existing, so a release that changes the
  native build without bumping its version would silently keep the old wheels
  on PyPI; that is refused against the previous tag.
- Every other torchsx dependency must have a release on PyPI that satisfies it,
  or users could not install the release.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
import urllib.request
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]
# What goes into the torchsx-native wheel; docs and the smoke test do not.
NATIVE_BUILD = ["native/src", "native/Cargo.toml", "native/Cargo.lock", "native/pyproject.toml"]


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def protocol(path: str, pattern: str) -> int:
    m = re.search(pattern, (ROOT / path).read_text(), re.M)
    if not m:
        sys.exit(f"no PROTOCOL in {path}")
    return int(m.group(1))


def on_pypi(req: Requirement) -> bool:
    """Whether PyPI has a non-yanked release satisfying ``req``."""
    with urllib.request.urlopen(f"https://pypi.org/pypi/{req.name}/json", timeout=30) as r:
        releases = json.load(r)["releases"]
    return any(
        files and not all(f.get("yanked") for f in files) and req.specifier.contains(v, prereleases=True)
        for v, files in releases.items()
    )


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
    errors = []

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    torchsx_version = Version(project["version"])
    native_version = Version(tomllib.loads((ROOT / "native/Cargo.toml").read_text())["package"]["version"])
    pins = [Requirement(r) for r in project["dependencies"] if Requirement(r).name == "torchsx-native"]
    print(f"torchsx {torchsx_version}, torchsx-native {native_version}, pin {[str(p) for p in pins]}")

    if tag is not None and tag != f"v{torchsx_version}":
        errors.append(f"tag {tag} does not match torchsx version v{torchsx_version}")
    module = re.search(r'^__version__ = "([^"]+)"$', (ROOT / "torchsx/__init__.py").read_text(), re.M)
    if not module or Version(module.group(1)) != torchsx_version:
        errors.append(f"torchsx/__init__.py __version__ does not match pyproject.toml's {torchsx_version}")
    if len(pins) != 1:
        errors.append("torchsx must depend on torchsx-native exactly once")
    elif native_version not in pins[0].specifier:
        errors.append(f"pin {pins[0]} does not accept native/Cargo.toml's {native_version}")

    for req in (Requirement(r) for r in project["dependencies"]):
        if req.name != "torchsx-native" and not on_pypi(req):
            errors.append(f"no release on PyPI satisfies {req}")

    rust = protocol("native/src/lib.rs", r"^const PROTOCOL: u32 = (\d+);")
    python = protocol("torchsx/io/_native.py", r"^PROTOCOL = (\d+)$")
    if rust != python:
        errors.append(f"PROTOCOL differs: native/src/lib.rs {rust}, torchsx/io/_native.py {python}")

    ref = tag or "HEAD"
    prev = git("describe", "--tags", "--abbrev=0", f"{ref}^")
    if prev.returncode == 0:
        prev_tag = prev.stdout.strip()
        old = git("show", f"{prev_tag}:native/Cargo.toml")
        if old.returncode == 0:
            old_version = Version(tomllib.loads(old.stdout)["package"]["version"])
            changed = git("diff", "--quiet", prev_tag, ref, "--", *NATIVE_BUILD).returncode != 0
            print(f"previous tag {prev_tag}: torchsx-native {old_version}, native build changed: {changed}")
            if changed and native_version <= old_version:
                errors.append(
                    f"native/ changed since {prev_tag} but torchsx-native is still {native_version}; "
                    "bump native/Cargo.toml (and the pin in pyproject.toml if the minor changes)"
                )
    else:
        print("no previous tag: first release")

    if errors:
        sys.exit("release check failed:\n  " + "\n  ".join(errors))
    print("release check passed")


if __name__ == "__main__":
    main()
