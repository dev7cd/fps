# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Blocking audit of the spherical pores (validation.geometric_audit.audit_pores) and the
pore-fiber test of the porosity generator, periodic images of the fibers included.
"""
import json

import numpy as np
import pytest

import main
import validation.geometric_audit as ga
from core.config import FiberPackingConfig
from core.fiber import Fiber
from core.void import Void
from generation.porosity_gen import PorosityGenerator

BOX = (1.0, 1.0, 1.0)
CFG = {'fiber_section_type': 'circular', 'section_parameters': {}}


def fiber(fid, pts, r=0.05):
    return Fiber(fid, np.array(pts, dtype=float), r, dict(CFG))


def straight_fibers():
    return [fiber(1, [(0.2, 0.2, 0.5), (0.2, 0.8, 0.5)]), fiber(2, [(0.6, 0.2, 0.5), (0.6, 0.8, 0.5)])]


def crossing_fiber():
    """Fiber along x crossing the face x = 1: its image shifted by -1 period enters near x = 0."""
    return [fiber(1, [(0.5, 0.5, 0.5), (1.08, 0.5, 0.5)])]


def test_no_pore_passes():
    res = ga.audit_pores(straight_fibers(), [], BOX)
    assert res['status'] == 'passed' and res['n_pores'] == 0


def test_disjoint_pores_pass_with_gaps():
    voids = [Void(1, np.array([0.4, 0.5, 0.2]), 0.05), Void(2, np.array([0.4, 0.5, 0.8]), 0.05)]
    res = ga.audit_pores(straight_fibers(), voids, BOX)
    assert res['status'] == 'passed' and res['n_pores'] == 2
    assert res['min_pore_pore_gap'] == pytest.approx(0.3)   # 0.4 apart through the periodic face
    assert res['min_pore_fiber_gap'] is None and 'within reach' in res['min_pore_fiber_gap_cause']


def test_overlapping_pores_fail():
    voids = [Void(1, np.array([0.4, 0.5, 0.2]), 0.05), Void(2, np.array([0.4, 0.5, 0.28]), 0.05)]
    res = ga.audit_pores(straight_fibers(), voids, BOX)
    assert res['status'] == 'failed' and res['responsible'] == {'pores': [1, 2]}


def test_pores_overlapping_across_a_face_fail():
    voids = [Void(1, np.array([0.4, 0.5, 0.03]), 0.05), Void(2, np.array([0.4, 0.5, 0.96]), 0.05)]
    assert ga.audit_pores(straight_fibers(), voids, BOX)['status'] == 'failed'


def test_pore_overlapping_a_fiber_fails():
    voids = [Void(1, np.array([0.2, 0.5, 0.58]), 0.05)]
    res = ga.audit_pores(straight_fibers(), voids, BOX)
    assert res['status'] == 'failed' and res['responsible']['fiber'] == 1


def test_pore_overlapping_a_periodic_image_of_a_fiber_fails():
    voids = [Void(1, np.array([0.06, 0.5, 0.5]), 0.03)]
    res = ga.audit_pores(crossing_fiber(), voids, BOX)
    assert res['status'] == 'failed'
    assert res['responsible']['fiber_shift_periods'] == [-1, 0, 0]


def test_generator_refuses_a_pore_on_a_periodic_image_of_a_fiber():
    """Regression: the generator used to test the root fibers only."""
    cfg = FiberPackingConfig(box_dims=BOX, seed=0, fiber_radius=0.05, fiber_section_type='circular',
                             section_parameters={}, generate_porosity=True)
    gen = PorosityGenerator(cfg)
    fibers = crossing_fiber()
    data = [{'centerline': f.centerline, 'radius': f.radius, 'bbox_min': f.bbox[0], 'bbox_max': f.bbox[1]}
            for f in fibers]
    assert not gen._check_void_fiber_collision_fast([Void(1, np.array([0.06, 0.5, 0.5]), 0.03)], data)
    assert gen._check_void_fiber_collision_fast([Void(1, np.array([0.3, 0.5, 0.2]), 0.03)], data)


def test_inconsistent_images_are_undetermined():
    voids = [Void(1, np.array([0.4, 0.5, 0.2]), 0.05), Void(1, np.array([0.45, 0.5, 0.2]), 0.05)]
    assert ga.audit_pores(straight_fibers(), voids, BOX)['status'] == 'undetermined'


def test_finalize_audits_generated_pores(tmp_path):
    cfg = FiberPackingConfig(box_dims=BOX, seed=1, fiber_radius=0.05, fiber_section_type='circular',
                             section_parameters={}, generate_porosity=True, target_void_fraction=0.01,
                             void_radius_mean=0.03, void_radius_std=0.005, export_mesh=False,
                             compute_spatial_stats=False, voxel_resolution=16, min_clearance=0.0,
                             output_prefix=str(tmp_path / 'rve'))
    main.finalize(cfg, crossing_fiber() + [fiber(2, [(0.5, 0.2, 0.2), (0.5, 0.8, 0.2)])], 0.0)
    param = json.loads((tmp_path / 'rve_parametric.json').read_text())
    pores = param['validity']['pores']
    assert pores['status'] == 'passed' and pores['n_pores'] > 0
    assert 'pores' not in param['validity']['excluded_checks']
    assert len(param['voids']) >= pores['n_pores']


def test_finalize_refuses_exports_when_restored_pores_overlap(tmp_path):
    cfg = FiberPackingConfig(box_dims=BOX, seed=1, fiber_radius=0.05, fiber_section_type='circular',
                             section_parameters={}, generate_porosity=True, export_mesh=False,
                             compute_spatial_stats=False, voxel_resolution=16, min_clearance=0.0,
                             output_prefix=str(tmp_path / 'rve'))
    with pytest.raises(SystemExit) as exc:
        main.finalize(cfg, crossing_fiber(), 0.0, voids=[Void(1, np.array([0.06, 0.5, 0.5]), 0.03)])
    assert exc.value.code == 1
    rec = json.loads((tmp_path / 'rve_validity.json').read_text())
    assert rec['status'] == 'failed' and rec['stage'] == 'pores'
    assert not (tmp_path / 'rve_parametric.json').exists()
