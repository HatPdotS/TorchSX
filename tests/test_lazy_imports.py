"""The package and its commands import torch only once they compute: ``--help`` never does."""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = re.findall(r'^"(torchsx\.[\w.-]+)"\s*=\s*"([\w.]+):(\w+)"\s*$',
                     (ROOT / 'pyproject.toml').read_text(), flags=re.M)


def run_python(code):
    path = [str(ROOT), os.environ.get('PYTHONPATH', '')]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in path if p))
    return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, check=True).stdout


def test_every_console_script_is_checked():
    assert {s[0] for s in SCRIPTS} >= {'torchsx.train', 'torchsx.merge', 'torchsx.merge-stats',
                                       'torchsx.observation-stats'}


@pytest.mark.parametrize('command, module, func', SCRIPTS, ids=[s[0] for s in SCRIPTS])
def test_help_does_not_import_torch(command, module, func):
    out = run_python(
        'import sys\n'
        f'from {module} import {func}\n'
        'try:\n'
        f'    {func}(["--help"])\n'
        'except SystemExit as exit:\n'
        '    assert exit.code in (0, None), exit.code\n'
        'print("torch imported:", "torch" in sys.modules)\n')
    assert f'usage: {command}' in out
    assert out.rstrip().endswith('torch imported: False')


def test_package_exports_load_on_first_use():
    out = run_python(
        'import sys\n'
        'import torchsx\n'
        'print("after import:", "torch" in sys.modules)\n'
        'from torchsx import HKLIndex, SFXStream, SourceInput\n'
        'import torchsx.merging.hkl, torchsx.stream\n'
        'assert SFXStream is torchsx.stream.SFXStream and HKLIndex is torchsx.merging.hkl.HKLIndex\n'
        'assert torchsx.io.sources.SourceInput is SourceInput\n'
        'assert set(torchsx.__all__) <= set(dir(torchsx))\n'
        'try:\n'
        '    torchsx.no_such_name\n'
        'except AttributeError:\n'
        '    print("unknown name: AttributeError")\n')
    assert 'after import: False' in out and 'unknown name: AttributeError' in out
