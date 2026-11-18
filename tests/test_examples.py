# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
The parametric records of the two example RVEs of the article are shipped in examples/ and
still rebuild the geometry that passed the audit (same fibers, same pores), including on a
platform whose floating-point reductions round differently (macOS arm64).
"""
from pathlib import Path

import pytest

import regenerate
from cli import config_from_args
from core.fiber import Fiber
from validation import geometric_audit

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

CASES = [
    ("RVE_05_parametric.json", 26, 0),
    ("RVE_UD_30_parametric.json", 156, 242),
]


def rebuild(name):
    record = regenerate.load_record(EXAMPLES / name)
    config = config_from_args(regenerate.namespace_from_record(record, {}))
    return record, regenerate.fibers_from_record(record, config)


@pytest.mark.parametrize("name, n_fibers, n_pores", CASES)
def test_example_record_rebuilds_audited_geometry(name, n_fibers, n_pores):
    record, fibers = rebuild(name)
    assert record["validity"]["status"] == "passed"
    assert len(fibers) == n_fibers
    regenerate.check_same_geometry(record, fibers)

    voids = regenerate.voids_from_record(record)
    assert len({v.id for v in voids}) == n_pores


def scale_volumes(monkeypatch, factor):
    volume = Fiber.get_real_volume
    monkeypatch.setattr(Fiber, "get_real_volume", lambda self: volume(self) * factor)


def test_volume_rounded_by_another_platform_is_accepted(monkeypatch):
    record, fibers = rebuild("RVE_05_parametric.json")
    scale_volumes(monkeypatch, 1 + 4e-16)  # a few ulps, the order of the differences between CPUs
    rebuilt = geometric_audit.fibers_sha256(geometric_audit.microstructure_records(fibers))
    assert rebuilt != record["validity"]["fibers_sha256"]  # a hash of the rebuilt fibers would refuse it
    assert regenerate.check_same_geometry(record, fibers) == record["validity"]["fibers_sha256"]


def test_a_different_volume_is_refused(monkeypatch):
    record, fibers = rebuild("RVE_05_parametric.json")
    scale_volumes(monkeypatch, 1 + 1e-6)
    with pytest.raises(regenerate.RecordError, match="volume"):
        regenerate.check_same_geometry(record, fibers)
