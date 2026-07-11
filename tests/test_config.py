"""! @brief Quick Doxygen documentation for test module."""
"""
Tests unitaires pour core/config.py — FiberPackingConfig.
"""
import pytest
import numpy as np
from core.config import FiberPackingConfig


class TestFiberPackingConfigDefaults:
    """Checks the default configuration values."""

    def test_default_box_dims(self):
        cfg = FiberPackingConfig()
        assert cfg.box_dims == (1.0, 1.0, 1.0)

    def test_default_target_volume_fraction(self):
        cfg = FiberPackingConfig()
        assert cfg.target_volume_fraction == 0.45

    def test_default_section_type(self):
        cfg = FiberPackingConfig()
        assert cfg.fiber_section_type == 'superelliptical'

    def test_default_seed_is_none(self):
        cfg = FiberPackingConfig()
        assert cfg.seed is None

    def test_default_min_clearance(self):
        cfg = FiberPackingConfig()
        assert cfg.min_clearance == 0.0

    def test_default_porosity_enabled(self):
        cfg = FiberPackingConfig()
        assert cfg.generate_porosity is True


class TestBoxVolume:
    """Checks the computation of the RVE domain volume."""

    def test_unit_cube(self):
        cfg = FiberPackingConfig(box_dims=(1.0, 1.0, 1.0))
        assert cfg.box_volume == pytest.approx(1.0)

    def test_rectangular_box(self):
        cfg = FiberPackingConfig(box_dims=(2.0, 3.0, 4.0))
        assert cfg.box_volume == pytest.approx(24.0)

    def test_small_box(self):
        cfg = FiberPackingConfig(box_dims=(0.1, 0.2, 0.5))
        assert cfg.box_volume == pytest.approx(0.01)


class TestEstimateRadius:
    """Checks the automatic estimation of the fiber radius."""

    def test_radius_auto_estimated_on_init(self):
        """Si fiber_radius est None, __post_init__ doit l'estimer."""
        cfg = FiberPackingConfig(fiber_radius=None)
        assert cfg.fiber_radius is not None
        assert cfg.fiber_radius > 0

    def test_radius_not_overwritten_if_set(self):
        """If fiber_radius is provided, it must not be modified."""
        cfg = FiberPackingConfig(fiber_radius=0.05)
        assert cfg.fiber_radius == pytest.approx(0.05)

    def test_radius_scales_with_box_size(self):
        """Un domaine plus grand devrait donner un rayon plus grand."""
        cfg_small = FiberPackingConfig(box_dims=(1.0, 1.0, 1.0), fiber_radius=None)
        cfg_large = FiberPackingConfig(box_dims=(10.0, 10.0, 10.0), fiber_radius=None)
        assert cfg_large.fiber_radius > cfg_small.fiber_radius

    def test_radius_positive_for_various_configs(self):
        """The estimated radius must always be positive."""
        for vf in [0.1, 0.3, 0.6]:
            cfg = FiberPackingConfig(
                target_volume_fraction=vf,
                fiber_radius=None,
            )
            assert cfg.fiber_radius > 0, f"Negative radius for Vf={vf}"


class TestSectionParameters:
    """Checks the default section parameters."""

    def test_default_section_params_keys(self):
        cfg = FiberPackingConfig()
        expected_keys = {'major_radius', 'minor_radius', 'exponent'}
        assert expected_keys == set(cfg.section_parameters.keys())

    def test_default_exponent(self):
        cfg = FiberPackingConfig()
        assert cfg.section_parameters['exponent'] == 2.5

    def test_custom_section_params(self):
        params = {'major_radius': 0.03, 'minor_radius': 0.02, 'exponent': 3.0}
        cfg = FiberPackingConfig(section_parameters=params)
        assert cfg.section_parameters['exponent'] == 3.0


class TestGenerationParameters:
    """Checks the CSAW generation parameters."""

    def test_default_orientation_bias(self):
        cfg = FiberPackingConfig()
        assert cfg.generation_parameters['orientation_bias'] == 'planar'

    def test_default_bias_strength(self):
        cfg = FiberPackingConfig()
        assert cfg.generation_parameters['bias_strength'] == pytest.approx(0.8)

    def test_max_curvature_angle(self):
        cfg = FiberPackingConfig()
        assert cfg.generation_parameters['max_curvature_angle'] == pytest.approx(np.pi / 6)
