# bench_local.py
import time
import numpy as np
from duopoly import p, P_MIN, P_MAX

def run(n_steps: int = 1000) -> None:
    rng = np.random.default_rng(0)  # stable run-to-run
    info = None
    prices_hist = np.zeros((2, 0))
    demand_hist = np.zeros((0,))

    my_last = (P_MIN + P_MAX) / 2.0
    comp_price = 50.0

    t0 = time.perf_counter()
    worst = 0.0

    for t in range(1, n_steps + 1):
        # feed last observation
        if t > 1:
            prices_hist = np.column_stack([prices_hist, [my_last, comp_price]])
            # simple synthetic demand: decreasing in our price, mildly increasing in competitor price
            d_hat = 20.0 - 0.15 * my_last + 0.05 * comp_price
            demand_hist = np.append(demand_hist, [max(0.0, d_hat)])

        # measure one call
        s = time.perf_counter()
        price, info = p(
            current_selling_season=1,
            selling_period_in_current_season=t,
            prices_historical_in_current_season=prices_hist,
            demand_historical_in_current_season=demand_hist,
            competitor_has_capacity_current_period_in_current_season=True,
            information_dump=info,
        )
        dt = time.perf_counter() - s
        if dt > worst:
            worst = dt

        my_last = float(price)
        # random walk competitor (kept within bounds)
        comp_price = float(np.clip(comp_price + rng.uniform(-2, 2), P_MIN, P_MAX))

    total = time.perf_counter() - t0
    print(f"Calls: {n_steps} | avg: {total/n_steps*1000:.3f} ms | max: {worst*1000:.3f} ms")

if __name__ == "__main__":
    run()
