# WRF Variable Mapping

Start with mappings that matter for Canary v0. Record WRF name, internal contract name, units, staggering, dimensions, and notes.

Initial placeholders:

| WRF | Contract | Units | Notes |
| --- | --- | --- | --- |
| T | perturbation_temperature | K | verify WRF semantics |
| U | u_wind | m s-1 | staggered |
| V | v_wind | m s-1 | staggered |
| W | vertical_velocity | m s-1 | staggered |
| QVAPOR | water_vapor_mixing_ratio | kg kg-1 | positivity check |
