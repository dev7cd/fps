# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Reproducibility: a seed reproduces the RVE bit for bit, the parametric record stores the
complete argument set, the pores and the software versions, and regenerate.py rebuilds every
export from the record without re-running placement.
"""
import filecmp
import json
import re
from pathlib import Path

import numpy as np
import pytest

import main
import regenerate
from cli import build_parser
from core.version import __version__

SMALL = ['--dims', '0.3', '0.3', '0.3', '--vf', '0.04', '--radius', '0.01', '--min_pts', '8',
         '--max_pts', '10', '--step', '0.02', '--porosity', '--void_vf', '0.005', '--void_mean', '0.012',
         '--res_fft', '24', '--no_mesh', '--no_spatial_stats']


def run(tmp_path, name, *extra):
    prefix = str(tmp_path / name)
    main.main(SMALL + ['--output', prefix] + list(extra))
    return prefix


def load(prefix):
    return json.loads(Path(f'{prefix}_parametric.json').read_text())


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('ref')
    return tmp, run(tmp, 'a', '--seed', '11')


def test_same_seed_same_rve(reference, tmp_path):
    _, a = reference
    b = run(tmp_path, 'b', '--seed', '11')
    A, B = load(a), load(b)
    assert A['microstructure_data'] == B['microstructure_data']
    assert A['voids'] == B['voids'] and len(A['voids']) > 0
    assert np.array_equal(np.load(f'{a}_voxelmap.npy'), np.load(f'{b}_voxelmap.npy'))


def test_record_holds_the_complete_argument_set(reference):
    _, a = reference
    rec = load(a)
    assert set(rec['cli_arguments']) == set(vars(build_parser().parse_args([])))
    assert rec['cli_arguments']['seed'] == 11
    assert rec['software']['version'] == __version__
    assert rec['validity']['status'] == 'passed' and rec['validity']['pores']['status'] == 'passed'


def test_unseeded_run_records_its_drawn_seed(tmp_path):
    rec = load(run(tmp_path, 'u'))
    assert isinstance(rec['cli_arguments']['seed'], int)
    assert rec['simulation_parameters']['random_seed'] == rec['cli_arguments']['seed']


def test_regeneration_is_bit_identical(reference):
    tmp, a = reference
    regenerate.main([f'{a}_parametric.json', '--output', str(tmp / 'g')])
    g = str(tmp / 'g')
    A, G = load(a), load(g)
    assert A['microstructure_data'] == G['microstructure_data'] and A['voids'] == G['voids']
    assert A['computed_statistics'] == G['computed_statistics']
    assert np.array_equal(np.load(f'{a}_voxelmap.npy'), np.load(f'{g}_voxelmap.npy'))
    for suffix in ('_roots.csv', '_full_periodic.csv', '_slice0000.pgm', '_slice0023.pgm'):
        assert filecmp.cmp(a + suffix, g + suffix, shallow=False)


def test_regeneration_with_another_export_resolution(reference):
    tmp, a = reference
    regenerate.main([f'{a}_parametric.json', '--res_fft', '40', '--output', str(tmp / 'g40')])
    assert np.load(str(tmp / 'g40_voxelmap.npy')).shape == (40, 40, 40)
    assert load(str(tmp / 'g40'))['microstructure_data'] == load(a)['microstructure_data']


def test_regeneration_refuses_geometry_options(reference):
    _, a = reference
    with pytest.raises(SystemExit) as exc:
        regenerate.main([f'{a}_parametric.json', '--vf', '0.5'])
    assert exc.value.code == 2


def test_regeneration_refuses_a_tampered_record(reference, tmp_path):
    _, a = reference
    rec = load(a)
    rec['microstructure_data'][0]['control_points'][0][0] += 1e-3
    bad = tmp_path / 'bad_parametric.json'
    bad.write_text(json.dumps(rec))
    with pytest.raises(SystemExit) as exc:
        regenerate.main([str(bad)])
    assert exc.value.code == 1


def test_version_matches_pyproject():
    text = (Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text()
    assert re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1) == __version__
