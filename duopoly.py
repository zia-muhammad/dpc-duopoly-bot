# duopoly.py
from typing import Any, Optional, Tuple, Union, List, Iterable
import numpy as np

P_MIN: float = 1.0
P_MAX: float = 100.0

# Keep only a small, recent window of observations to bound memory.
WINDOW: int = 50


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
            # not wrapped yet
            for i in range(self._count):
                yield self._buf[i]
            return
        # wrapped: start at write index, then to end, then from 0 to idx-1
        for i in range(self._idx, self._size):
            yield self._buf[i]
        for i in range(0, self._idx):
            yield self._buf[i]


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

    This is a minimal, platform compliant scaffold. It returns a deterministic
    mid-price and persists a tiny, bounded ring buffer of recent observations.
    Later steps will add learning and policy on top of this stable interface.
    """

    price: float = (P_MIN + P_MAX) / 2.0
    state: dict = information_dump if isinstance(information_dump, dict) else {}
    # Restore or create our small ring buffer.
    rb: RingBuffer = state.get("rb") if isinstance(state.get("rb"), RingBuffer) else RingBuffer(WINDOW)

    # If we are beyond period 1, record the last observed triple (my_price, comp_price, demand).
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
        except Exception:
            # Stay robust to unexpected shapes/values; we simply skip the update.
            pass

    # Persist minimal state for the next call.
    state["rb"] = rb
    state["version"] = 2  # simple internal state versioning

    return float(price), state
