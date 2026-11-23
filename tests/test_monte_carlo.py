import pytest

from monte_carlo import max_drawdown, run_monte_carlo


def test_drawdown_is_maximum_equity_peak_to_trough():
    assert max_drawdown([100.0, 90.0, 95.0, 80.0, 110.0]) == pytest.approx(0.2)


def test_monte_carlo_reproducible_with_seed():
    first = run_monte_carlo([1.0, -1.0], simulations=100, trades_per_run=50, seed=123)
    second = run_monte_carlo([1.0, -1.0], simulations=100, trades_per_run=50, seed=123)
    assert first == second


def test_monte_carlo_zero_trades_rejects():
    with pytest.raises(ValueError):
        run_monte_carlo([1.0], trades_per_run=0)


def test_never_trading_positive_returns_profitable_probability_is_zero_on_losses():
    report = run_monte_carlo([-1.0], simulations=100, trades_per_run=20)
    assert report["profitable_probability"] == 0.0
    assert report["ruin_probability"] == 1.0
