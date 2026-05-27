#!/usr/bin/env python
"""Emit the mountain-wave proof object from the high-priority runner."""

from __future__ import annotations

import sys

from pubtest_compare_ideal import main


if __name__ == "__main__":
    raise SystemExit(main(["--case", "schaer", *sys.argv[1:]]))
