import numpy as np
from duopoly import p, P_MIN, P_MAX

def test_p_returns_valid_tuple() -> None:
    price, info = p(
        current_selling_season=1,
        selling_period_in_current_season=1,
        prices_historical_in_current_season=None,
        demand_historical_in_current_season=None,
        competitor_has_capacity_current_period_in_current_season=True,
        information_dump=None,
    )
    assert isinstance(price, float)
    assert P_MIN <= price <= P_MAX
    assert isinstance(info, dict)
