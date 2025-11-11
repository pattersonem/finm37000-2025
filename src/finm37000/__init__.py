"""A package to support FINM37000."""

from .agg import (
    make_ohlcv as make_ohlcv,
)
from .continuous import (
    additive_splice as additive_splice,
    multiplicative_splice as multiplicative_splice,
)
from .db_env_util import (
    temp_env as temp_env,
    get_databento_api_key as get_databento_api_key,
)
from .futures import (
    favorite_def_cols as favorite_def_cols,
    get_all_legs_on as get_all_legs_on,
    get_official_stats as get_official_stats,
)
from .time import (
    as_ct as as_ct,
    get_cme_next_session_end as get_cme_next_session_end,
    get_cme_session_end as get_cme_session_end,
    tz_chicago as tz_chicago,
    us_business_day as us_business_day,
)


from .continuous import (
    get_roll_spec,
    constant_maturity_splice,
)

# If __all__ exists, append these names; otherwise create it.
try:
    __all__  # type: ignore[name-defined]
except NameError:
    __all__ = []

for _name in ("get_roll_spec", "constant_maturity_splice"):
    if _name not in __all__:
        __all__.append(_name)
