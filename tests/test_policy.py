import numpy as np
from duopoly import OnlineOLS3, choose_price, P_MIN, P_MAX

def test_choose_price_bounds_and_type() -> None:
    ols = OnlineOLS3(size=10)
    price = choose_price(
        ols=ols,
        competitor_price=50.0,
        competitor_has_capacity=True,
        last_price=None,
        n_obs=0,
    )
    assert isinstance(price, float)
    assert P_MIN <= price <= P_MAX
