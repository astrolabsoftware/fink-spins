import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def init_test(doctest_namespace):
    data_fink = "data/ztf/"

    # ZTF filters 1: g, 2: r
    filters = {"1": "g", "2": "r"}

    data_shg1g2 = pd.read_parquet(f"{data_fink}sso_fink_SHG1G2.parquet")  # All
    data_hg1g2 = pd.read_parquet(f"{data_fink}sso_fink_HG1G2.parquet")
    data_hg = pd.read_parquet(f"{data_fink}sso_fink_HG.parquet")

    doctest_namespace["data_hg1g2"] = data_hg1g2
    doctest_namespace["data_hg"] = data_hg
    doctest_namespace["data_shg1g2"] = data_shg1g2
    doctest_namespace["filters"] = filters
