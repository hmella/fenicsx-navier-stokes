"""
Write a copy of an example configuration with a shorter run length.

The examples take no parameters on the command line, so this is how `make examples` asks one of them for a handful of time steps instead of a full run. It copies the given YAML, overrides the keys in the ``Run:`` section that control length and output location, and writes the result somewhere disposable.

Usage:
    python examples/shorten_config.py examples/aorta/Ao11mmrest.yaml output/smoke_aorta.yaml \\
        --max-steps 3 --output output/smoke_aorta
"""

import argparse

import yaml


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("source", help="configuration file to copy")
    p.add_argument("dest", help="where to write the shortened copy")
    p.add_argument("--max-steps", type=int, required=True, help="value for Run.MaxSteps")
    p.add_argument("--output", required=True, help="value for Run.Output")
    args = p.parse_args(argv)

    with open(args.source) as fh:
        config = yaml.safe_load(fh)

    config.setdefault("Run", {})
    config["Run"]["MaxSteps"] = args.max_steps
    config["Run"]["Output"] = args.output
    config["Run"]["StoreAfter"] = None
    config["Run"]["Verbose"] = False

    with open(args.dest, "w") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    return args.dest


if __name__ == "__main__":
    main()
