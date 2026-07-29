import pandas as pd
import pytest

from src.data.loader import LoadedTable
from src.data.sql_store import SQLStore
from src.tools.sql_tool import is_read_only, run_sql


@pytest.fixture
def store():
    s = SQLStore()
    df = pd.DataFrame({"region": ["N", "S", "N"], "revenue": [100, 50, 75]})
    s.add_table(LoadedTable(table_name="sales", original_filename="sales.csv", df=df))
    return s


def test_is_read_only_accepts_select():
    assert is_read_only("SELECT * FROM sales")
    assert is_read_only("  with t as (select 1) select * from t ")


def test_is_read_only_rejects_mutations():
    assert not is_read_only("DROP TABLE sales")
    assert not is_read_only("DELETE FROM sales")
    assert not is_read_only("UPDATE sales SET revenue = 0")
    assert not is_read_only("")


def test_run_sql_success(store):
    result = run_sql(store, "SELECT region, SUM(revenue) as total FROM sales GROUP BY region")
    assert result.ok
    assert result.row_count == 2
    assert set(result.df["region"]) == {"N", "S"}


def test_run_sql_rejects_mutation(store):
    result = run_sql(store, "DELETE FROM sales")
    assert not result.ok
    assert "read-only" in result.error.lower()


def test_run_sql_reports_bad_sql(store):
    result = run_sql(store, "SELECT nonexistent_col FROM sales")
    assert not result.ok
    assert result.error
