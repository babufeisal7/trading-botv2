"""
Unified Trading System — Python paper-trading and backtesting prototype.

This implementation intentionally has no live-broker integration.  It provides
mechanical gate/scoring models, position sizing, capital protection, paper
execution, and event-based CSV backtesting.

Not a trading recommendation or a claim of a profitable edge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, time, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Literal, Optional
import argparse
import csv
import json
import math


class GateFailure(ValueError):
    pass


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class Gate0Input:
    tier1_event_minutes: int = 999
    tier2_event_minutes: Optional[int] = None
    event_risk_override: bool = False


@dataclass(frozen=True)
class Gate1Input:
    volatility_compression: bool
    clear_accumulation: bool
    near_htf_support_resistance: bool


@dataclass(frozen=True)
class Gate2Input:
    accumulation: bool
    manipulation_displacement: bool
    manipulation_rejection: bool
    distribution_continuation: bool


@dataclass(frozen=True)
class Gate4Input:
    qm_present: bool
    liquidity_zone: bool
    order_block_unmitigated: bool
    fvg_aligned: bool
    bos_confirmed: bool


@dataclass(frozen=True)
class Gate5Input:
    smt_confluence: bool


@dataclass(frozen=True)
class PinBarScores:
    location: int = 0
    displacement: int = 0
    volatility: int = 0
    candle: int = 0
    time_session: int = 0

    def total(self) -> int:
        scores = (self.location, self.displacement, self.volatility, self.candle, self.time_session)
        if any(v < 0 for v in scores):
            raise GateFailure("Pin-bar scores cannot be negative")
        limits = (30, 25, 15, 20, 10)
        if any(v > maximum for v, maximum in zip(scores, limits)):
            raise GateFailure("Pin-bar score exceeds dimension maximum")
        return sum(scores)


@dataclass(frozen=True)
class AMDScore:
    market_environment: int
    accumulation_quality: int
    manipulation_strength: int
    distribution_confirmation: int
    session_timing: int
    smt_confluence: bool

    def total(self) -> int:
        components = (
            (self.market_environment, 20),
            (self.accumulation_quality, 20),
            (self.manipulation_strength, 25),
            (self.distribution_confirmation, 20),
            (self.session_timing, 10),
        )
        if any(value < 0 or value > maximum for value, maximum in components):
            raise GateFailure("AMD component is outside its permitted range")
        smt = 5 if self.smt_confluence else 0
        return sum(value for value, _ in components) + smt


@dataclass(frozen=True)
class Setup:
    instrument: str
    direction: Direction
    gate0: Gate0Input
    gate1: Gate1Input
    gate2: Gate2Input
    gate3: AMDScore
    gate4: Gate4Input
    gate5: Gate5Input
    pin_bar: PinBarScores


@dataclass(frozen=True)
class SizingResult:
    account_balance: float
    risk_percent: float
    stop_distance: float
    unit_value: float
    risk_amount: float
    position_size: float


@dataclass
class Trade:
    instrument: str
    direction: Direction
    entry: float
    stop: float
    target1: float
    target2: Optional[float]
    position_size: float
    risk_percent: float
    timestamp: str
    status: Literal["open", "closed"] = "open"
    exit_price: Optional[float] = None
    result_r: Optional[float] = None


@dataclass
class Account:
    balance: float
    phase: Literal[0, 1, 2] = 0
    journaled_trades: int = 0
    weekly_losses: int = 0
    consecutive_losses: int = 0
    peak_balance: Optional[float] = None
    high_water_drawdown: float = 0.0
    trading_halted: bool = False

    def __post_init__(self) -> None:
        self.peak_balance = self.balance if self.peak_balance is None else self.peak_balance

    def update_drawdown(self) -> float:
        if self.peak_balance is None:
            self.peak_balance = self.balance
        self.peak_balance = max(self.peak_balance, self.balance)
        self.high_water_drawdown = max(
            self.high_water_drawdown,
            (self.peak_balance - self.balance) / self.peak_balance,
        )
        if self.high_water_drawdown > 0.08:
            self.trading_halted = True
            self.phase = 1
        return self.high_water_drawdown


def gate0_passes(value: Gate0Input) -> bool:
    if value.event_risk_override:
        return False
    if value.tier1_event_minutes <= 45:
        return False
    if value.tier2_event_minutes is not None:
        if 10 <= value.tier2_event_minutes <= 20:
            return False
    return True


def gate1_passes(value: Gate1Input) -> bool:
    return value.volatility_compression and value.clear_accumulation and value.near_htf_support_resistance


def gate2_passes(value: Gate2Input) -> bool:
    return value.accumulation and value.manipulation_displacement and value.manipulation_rejection and value.distribution_continuation


def gate4_passes(value: Gate4Input) -> bool:
    return value.qm_present and value.liquidity_zone and value.order_block_unmitigated and value.fvg_aligned and value.bos_confirmed


def amd_grade(score: int) -> tuple[Optional[str], Optional[float]]:
    if score >= 80:
        return "A+", 0.01
    if score >= 70:
        return "A", 0.0075
    if score >= 60:
        return "B", 0.0025
    return None, None


def pin_bar_grade(score: int) -> Optional[str]:
    if score >= 85:
        return "A+"
    if score >= 75:
        return "A"
    if score >= 65:
        return "B"
    return None


def event_risk_size_multiplier(value: Gate0Input) -> float:
    return 0.5 if value.event_risk_override else 1.0


def apply_risk_ladder(account: Account, grade_risk: float) -> float:
    if account.trading_halted:
        raise GateFailure("Trading halted by capital-protection protocol")
    if account.journaled_trades < 50:
        return min(0.005, grade_risk)
    if account.phase == 2:
        if grade_risk >= 0.01:
            return 0.015
        return grade_risk
    return grade_risk


def size_position(account: Account, stop_distance: float, unit_value: float, grade_risk: float, gate0: Gate0Input) -> SizingResult:
    if stop_distance <= 0:
        raise GateFailure("Stop distance must be positive")
    if unit_value <= 0:
        raise GateFailure("Unit value must be positive")
    if account.balance <= 0:
        raise GateFailure("Account balance must be positive")

    risk_percent = apply_risk_ladder(account, grade_risk)
    risk_percent *= event_risk_size_multiplier(gate0)
    risk_amount = account.balance * risk_percent
    position_size = risk_amount / (stop_distance * unit_value)

    return SizingResult(
        account_balance=account.balance,
        risk_percent=risk_percent,
        stop_distance=stop_distance,
        unit_value=unit_value,
        risk_amount=risk_amount,
        position_size=position_size,
    )


def validate_setup(setup: Setup) -> tuple[str, str, int, int]:
    if not gate0_passes(setup.gate0):
        raise GateFailure("Gate 0: event-risk filter failed")
    if not gate1_passes(setup.gate1):
        raise GateFailure("Gate 1: market-regime filter failed")
    if not gate2_passes(setup.gate2):
        raise GateFailure("Gate 2: AMD sequence failed")

    amd_score = setup.gate3.total()
    grade, grade_risk = amd_grade(amd_score)
    if grade is None or grade_risk is None:
        raise GateFailure("Gate 3: AMD score below B")

    if not gate4_passes(setup.gate4):
        raise GateFailure("Gate 4: QM/SMC confluence failed")
    if not setup.gate5.smt_confluence:
        # This is permitted: the Gate 3 SMT sub-score determines its effect.
        pass

    pin_score = setup.pin_bar.total()
    pin_grade = pin_bar_grade(pin_score)
    if pin_grade is None:
        raise GateFailure("Gate 6: pin-bar score below B")

    return grade, pin_grade, amd_score, pin_score


def session_for_timestamp(timestamp: datetime) -> str:
    utc = timestamp.astimezone(timezone.utc).time()
    if time(0, 0) <= utc < time(7, 0):
        return "asian"
    if time(7, 0) <= utc < time(10, 0):
        return "london"
    if time(12, 0) <= utc < time(16, 0):
        return "london_ny_overlap"
    if time(17, 0) <= utc:
        return "ny_afternoon_rollover"
    return "other"


def close_trade(account: Account, trade: Trade, exit_price: float) -> float:
    if trade.status != "open":
        raise GateFailure("Trade is already closed")
    risk_distance = abs(trade.entry - trade.stop)
    if risk_distance <= 0:
        raise GateFailure("Invalid trade risk distance")

    if trade.direction == Direction.LONG:
        pnl_r = (exit_price - trade.entry) / risk_distance
    else:
        pnl_r = (trade.entry - exit_price) / risk_distance

    trade.status = "closed"
    trade.exit_price = exit_price
    trade.result_r = pnl_r
    trade_value = trade.position_size * pnl_r * risk_distance
    account.balance += trade_value
    account.journaled_trades += 1

    if pnl_r < 0:
        account.consecutive_losses += 1
        account.weekly_losses += 1
    else:
        account.consecutive_losses = 0

    if account.consecutive_losses >= 2:
        account.trading_halted = True
    if account.weekly_losses >= 4:
        trade_risk = account.balance * trade.risk_percent
        account.phase = 1
    account.update_drawdown()
    return pnl_r


def expectancy(results_r: Iterable[float]) -> float:
    values = list(results_r)
    if not values:
        return 0.0
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    win_percent = len(wins) / len(values)
    loss_percent = len(losses) / len(values)
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    return win_percent * avg_win - loss_percent * avg_loss


def backtest_csv(csv_path: Path) -> dict:
    """Run a deliberately conservative event-level CSV backtest.

    Required columns:
      timestamp,instrument,direction,entry,stop,target1
    Optional: target2,unit_value,grade_risk,account_balance

    Each row is treated as an already-passed setup.  Use this as a sizing and
    journal engine; chart-derived gate detection should be added separately.
    """
    account = Account(balance=10_000.0)
    trades: list[Trade] = []

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "instrument", "direction", "entry", "stop", "target1"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames))
            raise ValueError(f"Missing CSV columns: {missing}")

        for row in reader:
            if account.trading_halted:
                break
            entry = float(row["entry"])
            stop = float(row["stop"])
            target1 = float(row["target1"])
            direction = Direction(row["direction"].lower())
            grade_risk = float(row.get("grade_risk") or 0.0025)
            unit_value = float(row.get("unit_value") or 1.0)
            stop_distance = abs(entry - stop)

            sizing = size_position(
                account,
                stop_distance,
                unit_value,
                grade_risk,
                Gate0Input(),
            )
            trade = Trade(
                instrument=row["instrument"],
                direction=direction,
                entry=entry,
                stop=stop,
                target1=target1,
                target2=float(row["target2"]) if row.get("target2") else None,
                position_size=sizing.position_size,
                risk_percent=sizing.risk_percent,
                timestamp=row["timestamp"],
            )

            exit_price = float(row["exit_price"]) if row.get("exit_price") else target1
            close_trade(account, trade, exit_price)
            trades.append(trade)

    results = [trade.result_r for trade in trades if trade.result_r is not None]
    report = {
        "ending_balance": round(account.balance, 2),
        "trades": len(trades),
        "expectancy_r": round(expectancy(results), 4),
        "max_drawdown": round(account.high_water_drawdown, 4),
        "trading_halted": account.trading_halted,
        "results": [
            {
                "timestamp": trade.timestamp,
                "instrument": trade.instrument,
                "direction": trade.direction.value,
                "result_r": trade.result_r,
            }
            for trade in trades
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Trading System paper/backtest engine")
    sub = parser.add_subparsers(dest="command", required=True)

    sizing = sub.add_parser("size", help="Calculate a position size")
    sizing.add_argument("--balance", type=float, required=True)
    sizing.add_argument("--stop-distance", type=float, required=True)
    sizing.add_argument("--unit-value", type=float, required=True)
    sizing.add_argument("--grade-risk", type=float, required=True)
    sizing.add_argument("--trades", type=int, default=0)

    backtest = sub.add_parser("backtest", help="Backtest an event-level CSV")
    backtest.add_argument("csv", type=Path)

    args = parser.parse_args()
    if args.command == "size":
        account = Account(balance=args.balance, journaled_trades=args.trades)
        result = size_position(account, args.stop_distance, args.unit_value, args.grade_risk, Gate0Input())
        print(json.dumps(asdict(result), indent=2))
    else:
        print(json.dumps(backtest_csv(args.csv), indent=2))


if __name__ == "__main__":
    main()
