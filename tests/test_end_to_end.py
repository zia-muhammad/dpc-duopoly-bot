import numpy as np
from duopoly import p

def test_end_to_end_progression() -> None:
    info = None
    prices_hist = np.zeros((2, 0))
    demand_hist = np.zeros((0,))
    my_prev = 50.0
    comp_prices = [50, 48, 49, 47, 46]

    for t, c in enumerate(comp_prices, start=1):
        if t > 1:
            prices_hist = np.column_stack([prices_hist, [my_prev, c]])
            demand_hist = np.append(demand_hist, [10.0])
        price, info = p(
            current_selling_season=1,
            selling_period_in_current_season=t,
            prices_historical_in_current_season=prices_hist,
            demand_historical_in_current_season=demand_hist,
            competitor_has_capacity_current_period_in_current_season=True,
            information_dump=info,
        )
        assert 1.0 <= price <= 100.0
        my_prev = price
