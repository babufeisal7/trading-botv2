# Unified Trading System v2.0

A Python paper-trading and backtesting implementation of the supplied gate-based
system. It contains no live-broker API and does not place real orders.

## Features

- Gates 0–6 validation primitives
- AMD and pin-bar scoring
- Position sizing from account balance, stop distance, and unit value
- Calibration, Phase 1, and Phase 2 risk-ladder hooks
- 2-loss/4-loss/drawdown protection hooks
- Expectancy and journalable event reports
- CSV backtesting for pre-qualified trade rows

## Important limitations

- Gate detection is not automated; chart features must be supplied by the caller.
- News blackout handling is not connected to an economic-calendar feed.
- SMT, OB, FVG, BOS, CHoCH, and liquidity checks remain explicit booleans.
- Correlated portfolio-heat across multiple open positions is not yet implemented.
- The prototype does not claim a positive, validated edge.

## Run

```bash
python unified_trading_system.py size --balance 10000 --stop-distance 40 --unit-value 10 --grade-risk 0.01
pytest -q
python unified_trading_system.py backtest example_events.csv
```

## Example CSV

```csv
timestamp,instrument,direction,entry,stop,target1,target2,exit_price,grade_risk,unit_value
2026-01-01T08:00:00Z,EUR/USD,long,1.1000,1.0950,1.1100,1.1200,1.1100,0.0075,1
```

Any `exit_price` is used for the prototype close; omit it to close at target1.
