"""
Tests unitaires pour generation/porosity_gen.py — PorosityGenerator.
"""
import pytest
import numpy as np
from core.config import FiberPackingConfig
from generation.porosity_gen import PorosityGenerator


class TestPorosityGeneratorDisabled:

    def test_no_voids_when_disabled(self):
        cfg = FiberPackingConfig(generate_porosity=False)
        gen = PorosityGenerator(cfg)
        voids = gen.generate_voids([])
        assert voids == []


class TestPorosityGeneratorEnabled:

    def _make_config(self, **kwargs):
        defaults = dict(
            box_dims=(1.0, 1.0, 1.0),
            generate_porosity=True,
            target_void_fraction=0.005,
            void_radius_mean=0.03,
            void_radius_std=0.005,
            seed=42,
            fiber_radius=0.02,
        )
        defaults.update(kwargs)
        return FiberPackingConfig(**defaults)

    def test_generates_voids(self):
        cfg = self._make_config()
        gen = PorosityGenerator(cfg)
        voids = gen.generate_voids([])
        assert len(voids) > 0

    def test_voids_inside_domain(self):
        cfg = self._make_config()
        gen = PorosityGenerator(cfg)
        voids = gen.generate_voids([])
        for v in voids:
            # Le centre peut déborder légèrement (images périodiques),
            # mais doit rester dans [-r, L+r]
            for i in range(3):
                assert v.center[i] >= -v.radius - 0.01
                assert v.center[i] <= cfg.box_dims[i] + v.radius + 0.01

    def test_voids_no_mutual_overlap(self):
        """Les voids ne doivent pas se chevaucher entre eux."""
        cfg = self._make_config(target_void_fraction=0.003)
        gen = PorosityGenerator(cfg)
        voids = gen.generate_voids([])
        for i in range(len(voids)):
            for j in range(i + 1, len(voids)):
                dist = np.linalg.norm(voids[i].center - voids[j].center)
                # Deux voids du même groupe (images périodiques) ont le même id
                if voids[i].id == voids[j].id:
                    continue
                assert dist >= voids[i].radius + voids[j].radius - 1e-6, \
                    f"Overlap entre void {i} et {j}: dist={dist:.4f}, seuil={voids[i].radius + voids[j].radius:.4f}"

    def test_reproducibility(self):
        cfg = self._make_config(seed=123)
        gen1 = PorosityGenerator(cfg)
        gen2 = PorosityGenerator(self._make_config(seed=123))
        v1 = gen1.generate_voids([])
        v2 = gen2.generate_voids([])
        assert len(v1) == len(v2)
        for a, b in zip(v1, v2):
            np.testing.assert_array_equal(a.center, b.center)

    def test_positive_radii(self):
        cfg = self._make_config()
        gen = PorosityGenerator(cfg)
        voids = gen.generate_voids([])
        for v in voids:
            assert v.radius > 0
