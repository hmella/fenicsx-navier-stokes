"""
YAML configuration loading with dot-notation access and optional physical units.

Any nested mapping carrying both a 'value' and a 'unit' key becomes a pint Quantity; everything else stays a plain scalar, list or nested DotDict.
"""

import yaml
from pint import Quantity as Q_


# Dictionary with attribute access
class DotDict(dict):
    """
    Dictionary whose keys are also reachable as attributes, so that ``pars.Problem.Viscosity`` works instead of ``pars["Problem"]["Viscosity"]``.
    """

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(attr) from None

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        try:
            del self[attr]
        except KeyError:
            raise AttributeError(attr) from None


# Strip units from a configuration value
def magnitude(value, unit=None):
    """
    Return a plain float from either a bare number or a pint Quantity, converting to 'unit' first when one is given. Use this wherever a value is handed to PETSc, so that both the bare and the unit-carrying spelling of a parameter work.
    """
    if isinstance(value, Q_):
        return float(value.to(unit).magnitude) if unit is not None else float(value.magnitude)
    return float(value)


# Parameter file reader
class ParameterHandler:
    """
    Load a YAML parameter file and expose it through dot notation.
    """

    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        self.params = self._load_and_parse()

    def _load_and_parse(self):
        with open(self.yaml_path) as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raise ValueError(f"{self.yaml_path}: file is empty")
        return self._parse_dict(raw)

    @classmethod
    def _parse_dict(cls, d):
        parsed = DotDict()
        for k, v in d.items():
            if isinstance(v, dict):
                # A {value, unit} pair becomes a physical quantity, anything else recurses
                if "value" in v and "unit" in v:
                    parsed[k] = Q_(v["value"], v["unit"])
                else:
                    parsed[k] = cls._parse_dict(v)
            elif isinstance(v, list):
                parsed[k] = [cls._parse_dict(i) if isinstance(i, dict) else i for i in v]
            else:
                parsed[k] = v
        return parsed

    def __getattr__(self, attr):
        # Dunder and private lookups are refused rather than delegated to self.params, which would recurse: copy.deepcopy probes __deepcopy__ on a fresh instance, before 'params' exists
        if attr.startswith("_") or attr == "params":
            raise AttributeError(attr)
        return getattr(self.params, attr)

    def __contains__(self, key):
        return key in self.params

    def __repr__(self):
        return f"ParameterHandler({self.yaml_path!r}, keys={sorted(self.params)})"
