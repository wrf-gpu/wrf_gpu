# NS01 native REAL Noah-MP snow gate

`snow_r4_savepoints.txt` = `proofs/noahmp/oracle/snow_oracle.f90` (verbatim WRF snow
routines) built at WRF's own REAL kind 4:

    gfortran -O2 -ffree-line-length-none proofs/noahmp/oracle/snow_oracle.f90 -o snow_oracle_r4
    ./snow_oracle_r4 snow_r4_savepoints.txt

The same source with `-fdefault-real-8 -fdefault-double-8` reproduces
`proofs/noahmp/fixtures/snow_oracle_savepoints.txt` byte-exactly (R8 reference).
Rule registered before measuring: `REGISTRATION.json`.
