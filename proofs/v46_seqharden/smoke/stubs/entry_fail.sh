#!/bin/bash
touch "${SMK_MARK:?}/entry_fail.started"
echo "entry stub: preflight"
JAX_PLATFORMS=cpu python3 -c 'raise ValueError("SCREEN6 prior budget ledger pin mismatch")'
