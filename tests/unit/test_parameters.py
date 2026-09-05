"""YAML configuration loading."""

import copy

import pytest
from pint import Quantity

from fenicsx_navier_stokes.parameters import DotDict, ParameterHandler, magnitude

CONFIGS = ["Ao9mmrest.yaml", "Ao11mmrest.yaml", "Ao13mmrest.yaml"]


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "Problem:\n"
        "  Viscosity: 0.04\n"
        "  Radius: {value: 4.0, unit: millimeter}\n"
        "  Windkessels:\n"
        "    Rd: [1.0, 2.0]\n"
        "Geometry:\n"
        "  OutletIDs: [6, 9]\n"
    )
    return ParameterHandler(path)


class TestDotDict:
    def test_attribute_access(self):
        assert DotDict({"a": 1}).a == 1

    def test_missing_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            _ = DotDict({}).nope

    def test_set_and_delete(self):
        d = DotDict()
        d.x = 5
        assert d["x"] == 5
        del d.x
        assert "x" not in d


class TestParameterHandler:
    def test_scalar_stays_plain(self, cfg):
        assert cfg.Problem.Viscosity == 0.04
        assert not isinstance(cfg.Problem.Viscosity, Quantity)

    def test_units_are_promoted(self, cfg):
        r = cfg.Problem.Radius
        assert isinstance(r, Quantity)
        assert r.to("cm").magnitude == pytest.approx(0.4)

    def test_nested_and_lists(self, cfg):
        assert cfg.Problem.Windkessels.Rd == [1.0, 2.0]
        assert cfg.Geometry.OutletIDs == [6, 9]

    def test_missing_key(self, cfg):
        with pytest.raises(AttributeError):
            _ = cfg.Nope

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ParameterHandler(tmp_path / "absent.yaml")

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            ParameterHandler(path)

    def test_deepcopy(self, cfg):
        """``copy.deepcopy`` must work.

        Delegating every attribute lookup to ``self.params`` made deepcopy recurse without bound: it probes ``__deepcopy__`` on a fresh instance, which lands in ``__getattr__`` before ``params`` exists, which looks up ``params``, and so on.
        """
        clone = copy.deepcopy(cfg)
        assert clone.Problem.Viscosity == cfg.Problem.Viscosity

    def test_contains_and_repr(self, cfg):
        assert "Problem" in cfg
        assert "Problem" in repr(cfg)


class TestMagnitude:
    def test_plain_number(self):
        assert magnitude(0.04) == 0.04

    def test_quantity(self):
        assert magnitude(Quantity(4.0, "millimeter"), "cm") == pytest.approx(0.4)

    def test_quantity_without_target_unit(self):
        assert magnitude(Quantity(4.0, "millimeter")) == pytest.approx(4.0)


@pytest.mark.parametrize("name", CONFIGS)
def test_shipped_configs_parse(repo_root, name):
    """Every shipped aorta configuration must load and carry the keys the solver reads."""
    pars = ParameterHandler(repo_root / "examples" / "aorta" / name)
    assert pars.Problem.Density > 0
    assert pars.Problem.Viscosity > 0
    assert isinstance(pars.Geometry.WallID, int)
    assert isinstance(pars.Geometry.InletID, int)
    assert len(pars.Geometry.OutletIDs) > 0
    wk = pars.Problem.Windkessels
    assert len(wk.Rd) == len(wk.Rp) == len(wk.C) == len(pars.Geometry.OutletIDs)
    assert pars.Solver.Kind in ("iterative", "direct")
