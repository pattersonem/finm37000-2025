from __future__ import annotations
from typing import Iterable, Mapping
import pandas as pd

def constant_maturity_splice(
    symbol: str,
    roll_spec: list[dict[str, str]],
    raw_data: pd.DataFrame,
    *,
    date_col: str = "datetime",
    price_col: str = "close",
) -> pd.DataFrame:
    # TODO: Return a DataFrame with columns matching the test's expectation: ['datetime','pre_price','pre_id','pre_expiration', 'next_price','next_id','next_expiration','pre_weight', symbol]
    raise NotImplementedError("Implement constant_maturity_splice")