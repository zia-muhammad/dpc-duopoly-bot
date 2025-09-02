# DPC Duopoly Bot

Dynamic pricing bot for a duopoly market, implemented as part of a coding challenge.

## Approach

- **Bounded memory**:  
  A fixed-size `RingBuffer` keeps the most recent `(my_price, competitor_price, demand)` triples. This ensures memory never grows unbounded.

- **Online learning**:  
  An `OnlineOLS3` class maintains sufficient statistics to estimate the demand function  
  \[
  demand ≈ α + β·my\_price + γ·competitor\_price
  \]  
  updated in **O(1)** time per observation.

- **Policy**:  
  `choose_price(...)` decides the next price using:
  - Model exploitation (fit from OLS)  
  - Exploration (`ε`-greedy, shrinking with sample size)  
  - Stability damping to avoid large jumps  
  - Monopoly uplift when competitor has no capacity  
  - Hard clamping between `P_MIN` and `P_MAX`

- **Interface**:  
  The main entrypoint `p(...)` respects the platform’s required signature and returns `(price, state)` where `state` is an opaque dictionary carrying minimal history.

## Tuning

Key knobs can be adjusted in `duopoly.py`:

- `WINDOW` — history size for online learning (default 50)
- `DAMP` — smoothing factor (0–1, default 0.20)
- `UNDERCUT` — relative price before slope is reliable (default 0.98)
- `EXPLORE_BAND` — exploration band vs competitor price (0.85–1.15)
- `MONOPOLY_UPLIFT` — uplift when competitor has no capacity (default 1.10)

## Development

### Install
```bash
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
