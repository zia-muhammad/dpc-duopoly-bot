from typing import Any, Optional, Tuple, Union
import numpy as np


P_MIN: float = 1.0
P_MAX: float = 100.0


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

    This is a minimal, platform compliant scaffold. Later steps will replace the
    fixed price with learned logic while keeping the signature and state shape.
    """
    price: float = (P_MIN + P_MAX) / 2.0
    state: dict = information_dump if isinstance(information_dump, dict) else {}
    state.setdefault("version", 1)
    return float(price), state
