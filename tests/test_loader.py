import pytest

from src.data.loader import CSVValidationError, sanitize_table_name, validate_and_load_csv


def test_sanitize_table_name_basic():
    assert sanitize_table_name("Sales Q1 (final).csv", set()) == "sales_q1__final_"


def test_sanitize_table_name_dedupes():
    existing = {"sales"}
    assert sanitize_table_name("sales.csv", existing) == "sales_2"


def test_rejects_empty_file():
    with pytest.raises(CSVValidationError):
        validate_and_load_csv("empty.csv", b"", set())


def test_rejects_zero_row_file():
    with pytest.raises(CSVValidationError):
        validate_and_load_csv("headers_only.csv", b"a,b,c\n", set())


def test_rejects_duplicate_headers():
    with pytest.raises(CSVValidationError):
        validate_and_load_csv("dupe.csv", b"a,a,b\n1,2,3\n", set())


def test_loads_valid_csv():
    raw = b"region,revenue\nNorth,100\nSouth,50\n"
    loaded = validate_and_load_csv("sales.csv", raw, set())
    assert loaded.table_name == "sales"
    assert list(loaded.df.columns) == ["region", "revenue"]
    assert len(loaded.df) == 2


def test_column_names_normalized():
    raw = b"Region Name,Total Revenue ($)\nNorth,100\n"
    loaded = validate_and_load_csv("sales.csv", raw, set())
    assert list(loaded.df.columns) == ["region_name", "total_revenue"]
