"""Allow `python -m llm_sidecar` as well as the `llm-sidecar` script.

The console script only exists after an install; this works straight from a
checkout, which is what a contributor has.
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
