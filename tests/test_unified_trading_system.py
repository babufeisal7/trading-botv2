from dataclasses import replace
import csv
from pathlib import Path
import tempfile

import pytest

from unified_trading_system import (
    Account,
    AMDScore,
    Direction,
    Gate0Input,
    Gate1Input,
    Gate2Input,
    Gate4Input,
    Gate5Input,
    GateFailure,
    PinBarScores,
    Setup,
    amd_grade,
    backtest_csv,
    close_trade,
    expectancy,
    gate0_passes,
    pin_bar_grade,
    size_position,
    validate_setup,
)


def qualifying_setup() -> Setup:
    return Setup(
        instrument="EUR/USD",
        direction=Direction.LONG,
        gate0=Gate0Input(tier1_event_minutes=999),
        gate1=Gate1Input(True, True, True),
        gate2=Gate2Input(True, True, True, True),
        gate3=AMDScore(20, 20, 25, 20, 10, True),
        gate4=Gate4Input(True, True, True, True, True),
        gate5=Gate5Input(True),
        pin_bar=PinBarScores(location=30, displacement=25, volatility=15, candle=20, time_session=10),
    )


def test_qualifying_setup_scores_a_plus():
    setup = qualifying_setup()
    grade, pin_grade, amd_score, pin_score = validate_setup(setup)
    assert (grade, pin_grade, amd_score, pin_score) == ("A+", "A+", 100, 100)


def test_news_gate_blocks_tier1_window():
    assert not gate0_passes(Gate0Input(tier1_event_minutes=15))


def test_news_gate_can_pass_outside_window():
    assert gate0_passes(Gate0Input(tier1_event_minutes=999))


def test_failed_gate_blocks_setup():
    setup = replace(qualifying_setup(), gate1=Gate1Input(False, True, True))
    with pytest.raises(GateFailure, match="Gate 1"):
        validate_setup(setup)


def test_position_sizing_10000_account_40_units_stop():
    account = Account(balance=10_000, journaled_trades=100)
    result = size_position(account, 40, 10, 0.01, Gate0Input())
    assert result.position_size == pytest.approx(0.25)


def test_calibration_phase_caps_risk_at_half_percent():
    account = Account(balance=10_000, journaled_trades=49)
    result = size_position(account, 40, 10, 0.01, Gate0Input())
    assert result.risk_percent == 0.005


def test_event_override_reduces_risk_again():
    account = Account(balance=10_000, journaled_trades=100)
    result = size_position(
        account,
        40,
        10,
        0.01,
        Gate0Input(event_risk_override=True),
    )
    assert result.risk_percent == 0.005


def test_amd_and_pin_grades():
    assert amd_grade(80) == ("A+", 0.01)
    assert amd_grade(70) == ("A", 0.0075)
    assert amd_grade(60) == ("B", 0.0025)
    assert pin_bar_grade(85) == "A+"
    assert pin_bar_grade(65) == "B"


def test_expectancy_formula():
    assert expectancy([1.0, 1.0, -1.0, -1.0]) == 0.0


def test_csv_backtest_runs_without_live_broker():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "events.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "instrument", "direction", "entry", "stop", "target1", "exit_price", "grade_risk", "unit_value"])
            writer.writerow(["2026-01-01T08:00:00Z", "EUR/USD", "long", "1.1000", "1.0950", "1.1100", "1.1100", "0.0075", "1"])
        report = backtest_csv(path)
        assert report["trades"] == 1
        assert report["ending_balance"] > 10_000
