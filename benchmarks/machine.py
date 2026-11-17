# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Description of the machine and software stack a benchmark ran on."""
import os
import platform
import subprocess
from pathlib import Path


def _cpu_model():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _memory_gb():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemTotal'):
                return round(int(line.split()[1]) / 1024 ** 2, 1)
    except OSError:
        pass
    return None


def machine_record(**extra):
    import numpy, scipy, numba
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True,
                                check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    rec = {'cpu': _cpu_model(), 'logical_cpus': os.cpu_count(), 'memory_gb': _memory_gb(),
           'os': platform.platform(), 'python': platform.python_version(), 'numpy': numpy.__version__,
           'scipy': scipy.__version__, 'numba': numba.__version__, 'fps_commit': commit}
    rec.update(extra)
    return rec
