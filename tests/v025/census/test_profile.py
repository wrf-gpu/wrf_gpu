import sqlite3

import pytest

from gpuwrf.diagnostics.census_profile import parse_profile


def profile_database(path, *, crossing=False, missing_on=False):
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,textId INTEGER);
        CREATE TABLE StringIds(id INTEGER,value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,copyKind INTEGER);
    """)
    for iteration in range(30):
        for offset, mode in ((0, "off"), (30000, "on")):
            if missing_on and mode == "on":
                continue
            start = iteration*100000+offset
            connection.execute("INSERT INTO NVTX_EVENTS VALUES(?,?,?,NULL)",
                               (start, start+25000, f"TSL:CENSUS:tree:{mode}:{iteration}"))
            connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?)",
                                   [(start+5000,start+15000),(start+8000,start+18000)])
            if mode == "on":
                connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?)",
                                   (start+18100,start+(26000 if crossing else 18130)))
            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,?)",
                               (start+1,start+2,1))
    connection.commit()
    connection.close()


def test_profile_uses_union_and_sum_with_paired_intervals(tmp_path):
    path = tmp_path / "profile.sqlite"
    profile_database(path)
    result = parse_profile(path)
    assert result["metrics"]["kernel_sum_ns"]["off_mean_ns"] == 20000
    assert result["metrics"]["device_union_ns"]["off_mean_ns"] == 13000
    assert result["metrics"]["kernel_sum_ns"]["delta_mean_ns"] == 30
    assert result["metrics"]["kernel_sum_ns"]["pairs"] == 30
    assert result["metrics"]["all_device_sum_ns"]["off_mean_ns"] == 20001
    assert result["metrics"]["all_device_union_ns"]["off_mean_ns"] == 13001
    assert result["metrics"]["all_device_sum_ns"]["delta_mean_ns"] == 30
    assert result["gate_ordinary_lt_1pct"]
    assert result["gate_all_device_lt_1pct"]
    assert result["ranges"][0]["copies_by_kind"] == {"1": 1}


@pytest.mark.parametrize("mutation", ["crossing", "missing_on"])
def test_profile_rejects_incomplete_or_crossing_device_ranges(tmp_path, mutation):
    path = tmp_path / "profile.sqlite"
    profile_database(path, **{mutation: True})
    with pytest.raises(ValueError):
        parse_profile(path)


@pytest.mark.parametrize("bad_order", [False, True])
def test_module_markers_recover_only_complete_predeclared_pairs(tmp_path, bad_order):
    path = tmp_path / "profile.sqlite"
    profile_database(path)
    connection = sqlite3.connect(path)
    connection.execute("DELETE FROM NVTX_EVENTS")
    connection.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL")
    connection.execute("DELETE FROM CUPTI_ACTIVITY_KIND_MEMCPY")
    for index in range(60):
        pair, within = divmod(index, 2)
        mode = ("off", "on")[within] if pair % 2 == 0 else ("on", "off")[within]
        program = 17 if mode == "off" else 18
        if bad_order and index == 10:
            program = 18 if program == 17 else 17
        start = index * 30000
        connection.execute("INSERT INTO NVTX_EVENTS VALUES(?,?,?,NULL)",
                           (start, start+9000, f"XlaModule:#hlo_module=jit_ordinary,program_id={program}#"))
        # GPU work finishes after the CPU module scope but before the next call.
        connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?)",
                               [(start+5000,start+15000),(start+8000,start+18000)])
        if mode == "on":
            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?)", (start+18100,start+18130))
    connection.commit()
    connection.close()
    if bad_order:
        with pytest.raises(ValueError, match="alternating pairs"):
            parse_profile(path)
    else:
        result = parse_profile(path)
        assert result["metrics"]["kernel_sum_ns"]["off_mean_ns"] == 20000
        assert result["metrics"]["kernel_sum_ns"]["delta_mean_ns"] == 30
        assert result["metrics"]["kernel_sum_ns"]["pairs"] == 30
        assert result["gate_ordinary_lt_1pct"]
