import numpy as np
from duopoly import RingBuffer, OnlineOLS3

def test_ring_buffer_rollover() -> None:
    rb = RingBuffer(size=3)
    rb.add((1,2,3)); rb.add((4,5,6)); rb.add((7,8,9)); rb.add((10,11,12))
    items = list(rb.items())
    assert len(items) == 3
    assert items[0] == (4.0, 5.0, 6.0)
    assert items[-1] == (10.0, 11.0, 12.0)

def test_online_ols_basic_fit() -> None:
    # demand = 5 - 0.2*p + 0.1*c + noise
    rng = np.random.default_rng(0)
    P = rng.uniform(5, 90, 40)
    C = rng.uniform(5, 90, 40)
    D = 5.0 - 0.2*P + 0.1*C + rng.normal(0, 0.3, 40)

    ols = OnlineOLS3(size=50)
    for p, c, d in zip(P, C, D):
        ols.add(float(p), float(c), float(d))

    coefs = ols.coeffs()
    assert coefs is not None
    a, b, g = coefs
    # slope near -0.2
    assert -0.3 < b < -0.1
    assert  0.05 < g <  0.15
