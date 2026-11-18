# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
The parametric records of the two example RVEs of the article are shipped in examples/ and
still rebuild the geometry that passed the audit (same fibers, same pores).
"""
from pathlib import Path

import pytest

import regenerate
from cli import config_from_args

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

CASES = [
    ("RVE_05_parametric.json", 26, 0),
    ("RVE_UD_30_parametric.json", 156, 242),
]


@pytest.mark.parametrize("name, n_fibers, n_pores", CASES)
def test_example_record_rebuilds_audited_geometry(name, n_fibers, n_pores):
    record = regenerate.load_record(EXAMPLES / name)
    assert record["validity"]["status"] == "passed"

    config = config_from_args(regenerate.namespace_from_record(record, {}))
    fibers = regenerate.fibers_from_record(record, config)
    assert len(fibers) == n_fibers
    regenerate.check_same_geometry(record, fibers)

    voids = regenerate.voids_from_record(record)
    assert len({v.id for v in voids}) == n_pores
