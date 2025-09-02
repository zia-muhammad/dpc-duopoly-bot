from typing import Any, Optional, Tuple, Union, List, Iterable
import numpy as np

# Explicit price bounds.
P_MIN: float = 1.0
P_MAX: float = 100.0
# Keep only a small, recent window of observations to bound memory.
WINDOW: int = 50

DAMP: float = 0.20
# default relative price when model isn't reliable yet
UNDERCUT: float = 0.98
# ε-exploration band vs competitor price
EXPLORE_BAND: Tuple[float, float] = (0.85, 1.15)
# uplift when competitor has no capacity
MONOPOLY_UPLIFT: float = 1.10


class RingBuffer:
    """
    Fixed-size buffer for (my_price, comp_price, demand).

    - Append is O(1).
    - Memory is bounded by WINDOW.
    - Iteration yields items in chronological order (oldest → newest).
    """
    __slots__ = ("_size", "_buf", "_idx", "_count")

    def __init__(self, size: int = WINDOW) -> None:
        self._size = int(size)
        self._buf: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * self._size
        self._idx = 0
        self._count = 0

    def add(self, triple: Tuple[float, float, float]) -> None:
        p_my, p_comp, d = map(float, triple)
        self._buf[self._idx] = (p_my, p_comp, d)
        self._idx = (self._idx + 1) % self._size
        if self._count < self._size:
            self._count += 1

    def __len__(self) -> int:
        return self._count

    def items(self) -> Iterable[Tuple[float, float, float]]:
        """Yield items in chronological order (oldest → newest)."""
        if self._count < self._size:
            for i in range(self._count):
                yield self._buf[i]
            return
        for i in range(self._idx, self._size):
            yield self._buf[i]
        for i in range(0, self._idx):
            yield self._buf[i]


class OnlineOLS3:
    """
    Online OLS for: demand ~ alpha + beta*my_price + gamma*comp_price

    Maintains sufficient statistics so each add() is O(1). Memory is bounded via an
    internal RingBuffer of size WINDOW.
    """
    __slots__ = ("_rb", "_n", "_sp", "_sc", "_spp", "_spc", "_scc", "_sd", "_spd", "_scd")

    def __init__(self, size: int = WINDOW) -> None:
        self._rb = RingBuffer(size)
        self._n = 0.0
        self._sp = 0.0   # sum p
        self._sc = 0.0   # sum c
        self._spp = 0.0  # sum p^2
        self._spc = 0.0  # sum p*c
        self._scc = 0.0  # sum c^2
        self._sd = 0.0   # sum d
        self._spd = 0.0  # sum p*d
        self._scd = 0.0  # sum c*d

    def _accum(self, p: float, c: float, d: float, sgn: float) -> None:
        self._n += sgn
        self._sp += sgn * p
        self._sc += sgn * c
        self._spp += sgn * p * p
        self._spc += sgn * p * c
        self._scc += sgn * c * c
        self._sd += sgn * d
        self._spd += sgn * p * d
        self._scd += sgn * c * d

    def add(self, p: float, c: float, d: float) -> None:
        # Evict oldest if buffer full: subtract its contribution.
        if len(self._rb) == self._rb._size:
            # Oldest item is the first from items()
            op, oc, od = next(self._rb.items())
            self._accum(op, oc, od, -1.0)
        self._rb.add((p, c, d))
        self._accum(p, c, d, +1.0)

    def count(self) -> int:
        return len(self._rb)

    def coeffs(self) -> Optional[Tuple[float, float, float]]:
        """
        Solve (X'X) * theta = X'y for theta = [alpha, beta, gamma].
        Returns None if not enough data or matrix is ill-conditioned.
        """
        if self._n < 3:
            return None

        M = np.array(
            [
                [self._n,  self._sp,  self._sc],
                [self._sp, self._spp, self._spc],
                [self._sc, self._spc, self._scc],
            ],
            dtype=float,
        )
        v = np.array([self._sd, self._spd, self._scd], dtype=float)

        try:
            if np.linalg.cond(M) > 1e10:
                return None
            sol = np.linalg.solve(M, v)  # [alpha, beta, gamma]
            return float(sol[0]), float(sol[1]), float(sol[2])
        except Exception:
            return None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


EPS_MIN: float = 0.05  # exploration floor


def choose_price(
    ols: OnlineOLS3,
    competitor_price: Optional[float],
    competitor_has_capacity: bool,
    last_price: Optional[float],
    n_obs: int,
) -> float:
    """
    Compute a candidate price using the current OLS fit and light exploration.
    Pure function: does not mutate state.
    """
    # Cold start: no competitor info yet → midpoint.
    if competitor_price is None:
        return (P_MIN + P_MAX) / 2.0

    # Exploit current fit: d ≈ α + β·p + γ·c
    coefs = ols.coeffs()
    if coefs is not None:
        alpha, beta, gamma = coefs
        if beta < 0.0:
            p_star = - (alpha + gamma * float(competitor_price)) / (2.0 * beta)
        else:
            # Non-negative slope is suspicious → hug competitor.
            p_star = float(competitor_price)
    else:
        # Not enough data yet → gentle undercut.
        p_star = float(competitor_price) * UNDERCUT

    # If competitor is out of capacity, nudge toward a “monopoly-like” price.
    if competitor_price is not None and not competitor_has_capacity:
        p_star *= MONOPOLY_UPLIFT

    # ε-greedy exploration (decays with n).
    eps = max(EPS_MIN, 1.0 / np.sqrt(max(1, n_obs)))
    if np.random.random() < eps:
        lo = _clamp(float(competitor_price) * EXPLORE_BAND[0], P_MIN, P_MAX)
        hi = _clamp(float(competitor_price) * EXPLORE_BAND[1], P_MIN, P_MAX)
        p_star = float(np.random.uniform(lo, hi))

    # Bound + light damping for stability.
    p_star = _clamp(p_star, P_MIN, P_MAX)
    if last_price is not None:
        p_star = (1.0 - DAMP) * float(last_price) + DAMP * p_star

    return _clamp(p_star, P_MIN, P_MAX)


def p(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season: Union[np.ndarray, None],
    demand_historical_in_current_season: Union[np.ndarray, None],
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump: Optional[Any] = None,
) -> Tuple[float, Any]:
    """
    Return next price and an opaque state object.

    This is a minimal, platform compliant scaffold. It persists a small, bounded
    state (ring buffer + online OLS) and chooses a price using learned stats.
    """
    # Persistent state between calls.
    state: dict = information_dump if isinstance(information_dump, dict) else {}

    # Restore/create bounded structures.
    rb: RingBuffer = state.get("rb") if isinstance(state.get("rb"), RingBuffer) else RingBuffer(WINDOW)
    ols: OnlineOLS3 = state.get("ols") if isinstance(state.get("ols"), OnlineOLS3) else OnlineOLS3(WINDOW)

    # If beyond period 1, record the last observation (my_price, comp_price, demand).
    if (
        selling_period_in_current_season > 1
        and prices_historical_in_current_season is not None
        and demand_historical_in_current_season is not None
    ):
        try:
            my_last = float(prices_historical_in_current_season[0, -1])
            comp_last = float(prices_historical_in_current_season[1, -1])
            d_last = float(demand_historical_in_current_season[-1])
            if np.isfinite(d_last) and d_last >= 0.0:
                rb.add((my_last, comp_last, d_last))
                ols.add(my_last, comp_last, d_last)
        except Exception:
            # Skip malformed inputs gracefully.
            pass

    # Choose the next price using learned stats.
    competitor_price: Optional[float] = None
    if prices_historical_in_current_season is not None and selling_period_in_current_season > 1:
        try:
            competitor_price = float(prices_historical_in_current_season[1, -1])
        except Exception:
            competitor_price = None

    last_price: Optional[float] = state.get("last_price")

    price: float = choose_price(
        ols=ols,
        competitor_price=competitor_price,
        competitor_has_capacity=bool(competitor_has_capacity_current_period_in_current_season),
        last_price=last_price,
        n_obs=ols.count(),
    )

    # Persist minimal state for the next call.
    state["rb"] = rb
    state["ols"] = ols
    state["last_price"] = float(price)
    state["version"] = 4

    return float(price), state
