"""
Lightweight local simulator for duopoly.p(...).

- Simulates S seasons x T periods.
- Competitor price follows a clipped random walk.
- Our demand is a simple linear function of prices with noise.
- Histories are maintained in the shapes required by duopoly.p:
    prices_hist: shape (2, t-1) -> [ [my_prices...], [comp_prices...] ]
    demand_hist: shape (t-1, )
- Optional per-call timing via env var: TIMING=1
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Tuple

import numpy as np

# Import pricing policy
import duopoly as bot


@dataclass
class SimConfig:
    seasons: int = 3
    periods: int = 100
    demand_intercept: float = 20.0 
    own_price_slope: float = -0.15     
    comp_price_slope: float = 0.05     
    noise_std: float = 0.5
    comp_walk_step: float = 2.0        
    seed: int = 7
    timing: bool = False           


def _synthetic_demand(p_my: float, p_comp: float, cfg: SimConfig, rng: np.random.Generator) -> float:
    """Toy demand function with Gaussian noise, clamped at 0."""
    d_hat = cfg.demand_intercept + cfg.own_price_slope * p_my + cfg.comp_price_slope * p_comp
    noise = rng.normal(0.0, cfg.noise_std)
    return float(max(0.0, d_hat + noise))


def run_sim(cfg: SimConfig) -> None:
    rng = np.random.default_rng(cfg.seed)

    all_revenues = []
    timings = []

    for season in range(1, cfg.seasons + 1):
        # Reset state at the start of each season
        information_dump: Optional[dict] = None
        prices_hist = np.zeros((2, 0), dtype=float)  # [my; comp]
        demand_hist = np.zeros((0,), dtype=float)

        # Initialize last prices
        my_last = (bot.P_MIN + bot.P_MAX) / 2.0
        comp_last = float(np.clip(50.0 + rng.uniform(-5, 5), bot.P_MIN, bot.P_MAX))

        season_revenue = 0.0

        for t in range(1, cfg.periods + 1):
            # Feed last observation (for t>1)
            if t > 1:
                prices_hist = np.column_stack([prices_hist, [my_last, comp_last]])
                # Realized demand from previous period (using prices of previous period)
                d_last = _synthetic_demand(my_last, comp_last, cfg, rng)
                demand_hist = np.append(demand_hist, d_last)

            # Whether competitor has capacity (toy rule: 95% chance has stock)
            comp_has_capacity = rng.random() < 0.95

            # Call the bot (time it if requested)
            if cfg.timing:
                t0 = perf_counter()
                price, information_dump = bot.p(
                    current_selling_season=season,
                    selling_period_in_current_season=t,
                    prices_historical_in_current_season=prices_hist,
                    demand_historical_in_current_season=demand_hist,
                    competitor_has_capacity_current_period_in_current_season=comp_has_capacity,
                    information_dump=information_dump,
                )
                timings.append(perf_counter() - t0)
            else:
                price, information_dump = bot.p(
                    current_selling_season=season,
                    selling_period_in_current_season=t,
                    prices_historical_in_current_season=prices_hist,
                    demand_historical_in_current_season=demand_hist,
                    competitor_has_capacity_current_period_in_current_season=comp_has_capacity,
                    information_dump=information_dump,
                )

            # Enforce bounds defensively
            price = float(np.clip(price, bot.P_MIN, bot.P_MAX))

            # Revenue realized for the *current* period uses the demand we will realize
            # with today's prices (my price chosen now and competitor price chosen now).
            # We advance competitor price as a random walk (bounded).
            comp_next = float(np.clip(comp_last + rng.uniform(-cfg.comp_walk_step, cfg.comp_walk_step),
                                      bot.P_MIN, bot.P_MAX))

            demand_today = _synthetic_demand(price, comp_next, cfg, rng)
            revenue_today = price * demand_today
            season_revenue += revenue_today

            # Roll forward
            my_last = price
            comp_last = comp_next

        all_revenues.append(season_revenue)

        print(f"Season {season:>2d} | periods={cfg.periods} | revenue={season_revenue:,.2f}")

    # Summary
    avg_rev = float(np.mean(all_revenues)) if all_revenues else 0.0
    std_rev = float(np.std(all_revenues)) if all_revenues else 0.0
    print("-" * 72)
    print(f"Seasons: {cfg.seasons} | Periods/season: {cfg.periods}")
    print(f"Revenue  | mean={avg_rev:,.2f} | std={std_rev:,.2f} | min={min(all_revenues):,.2f} | max={max(all_revenues):,.2f}")

    if cfg.timing and timings:
        timings = np.array(timings, dtype=float)
        avg_ms = timings.mean() * 1e3
        p95_ms = np.quantile(timings, 0.95) * 1e3
        worst_ms = timings.max() * 1e3
        print(f"[timing] calls={len(timings)} | avg={avg_ms:.3f} ms | p95={p95_ms:.3f} ms | max={worst_ms:.3f} ms")


def parse_args() -> SimConfig:
    parser = argparse.ArgumentParser(description="Local simulator for duopoly.p(...)")
    parser.add_argument("--seasons", type=int, default=3, help="Number of seasons to simulate (default: 3)")
    parser.add_argument("--periods", type=int, default=100, help="Number of periods per season (default: 100)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed (default: 7)")
    parser.add_argument("--alpha", type=float, default=20.0, help="Demand intercept α (default: 20.0)")
    parser.add_argument("--beta", type=float, default=-0.15, help="Own price slope β (<0) (default: -0.15)")
    parser.add_argument("--gamma", type=float, default=0.05, help="Competitor price slope γ (>0) (default: 0.05)")
    parser.add_argument("--noise", type=float, default=0.5, help="Demand noise std dev (default: 0.5)")
    parser.add_argument("--comp-step", type=float, default=2.0, help="Competitor random-walk step (default: 2.0)")
    cfg = SimConfig()
    args = parser.parse_args()

    timing_env = os.getenv("TIMING", "0") == "1"

    cfg.seasons = args.seasons
    cfg.periods = args.periods
    cfg.seed = args.seed
    cfg.demand_intercept = args.alpha
    cfg.own_price_slope = args.beta
    cfg.comp_price_slope = args.gamma
    cfg.noise_std = args.noise
    cfg.comp_walk_step = args.comp_step
    cfg.timing = timing_env
    return cfg


if __name__ == "__main__":
    config = parse_args()
    run_sim(config)
