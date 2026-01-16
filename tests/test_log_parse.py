from pathlib import Path

from skills.metrics_parse import parse_lammps_timing, parse_thermo_series, parse_thermo_table


def test_parse_lammps_timing():
    log_text = Path("examples/sample_lammps_log/log.lammps").read_text(encoding="utf-8")
    timing = parse_lammps_timing(log_text)
    assert timing["total"] == 1.2345
    assert timing["pair"] == 0.41
    assert timing["comm"] == 0.21


def test_parse_thermo_table():
    log_text = Path("examples/sample_lammps_log/log.lammps").read_text(encoding="utf-8")
    thermo = parse_thermo_table(log_text)
    assert thermo["Temp"] == 1.2
    assert thermo["TotEng"] == -4.1


def test_parse_thermo_series():
    log_text = Path("examples/sample_lammps_log/log.lammps").read_text(encoding="utf-8")
    series = parse_thermo_series(log_text, max_rows=10)
    assert series["Temp"] == [1.0, 1.2]
    assert series["TotEng"] == [-4.0, -4.1]
