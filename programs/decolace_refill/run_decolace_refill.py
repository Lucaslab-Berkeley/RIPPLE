"""Re-fill a DeCo-LACE sum outside its beam, and write its 2DTM search mask.

The pixels outside the (reduced) beam ellipse are replaced by Gaussian noise with the
beam interior's radial power spectrum, so a 2DTM search's whitening and noise model see
a stationary image. Contamination blobs inside the beam are left out of the search
mask. See :mod:`ripple.core.sum_refill`.

Usage::

    python run_decolace_refill.py decolace_refill_example_config.yaml

The beam comes from the DeCo-LACE pipeline's ``beam_mask_output_path`` JSON (the
ellipse fitted on the raw movie, with its crop), or from a mask image in the sum's
frame.
"""

import argparse
import json

import yaml

from ripple.config import RefillConfig
from ripple.managers import RefillManager

REFILL_YAML_PATH = "decolace_refill_example_config.yaml"


def main() -> None:
    """Re-fill the configured sum and write the outputs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", nargs="?", default=REFILL_YAML_PATH)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    manager = RefillManager(refill_config=RefillConfig(**config["refill_config"]))
    result = manager.run()
    print(json.dumps(result["stats"], indent=1))


if __name__ == "__main__":
    main()
