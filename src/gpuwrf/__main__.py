"""Enable ``python -m gpuwrf ...`` as an alias for the ``gpuwrf`` console script."""

from __future__ import annotations

import sys

from gpuwrf.cli import main

if __name__ == "__main__":
    sys.exit(main())
