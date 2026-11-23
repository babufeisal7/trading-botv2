import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import quantiles
from typing import Iterable, Sequence


@dataclass(frozen=True)
class MonteCarloResult:
    final_balance: float
    peak_balance: float
    max_drawdown: float
    minimum_balance: float
    ruin: bool
    halted: bool
    total_trades: int


def max_drawdown(equity: Sequence[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    worst = 0.0
    for balance in equity:
        peak = max(peak, balance)
        worst = max(worst, (peak - balance) / peak)
    return worst


def run_monte_carlo(
    observed_r: Sequence[float],
    *,
    simulations: int = 10_000,
    trades_per_run: int = 200,
    starting_balance: float = 10_000.0,
    calibration_trades: int = 49,
    risk_percent: float = 0.0075,
    ruin_threshold: float = 0.50,
    seed: int | None = 42,
) -> dict:
    """Bootstrap Monte Carlo: randomly reorder sampled historical R-multiples.

    This preserves the empirical R-multiple distribution and tests sequence risk.
    It does not assume trade independence beyond the bootstrap sample.
    """
    if not observed_r:
        raise ValueError("observed_r must contain at least one result")
    if any(not isinstance(value, (int, float)) for value in observed_r):
        raise ValueError("observed R-multiples must be numeric")
    if simulations < 1 or trades_per_run < 1:
        raise ValueError("simulations and trades_per_run must be positive")
    if starting_balance <= 0 or ruin_threshold <= 0:
        raise ValueError("starting_balance and ruin_threshold must be positive")

    rng = random.Random(seed)
    outcomes: list[MonteCarloResult] = []

    for _ in range(simulations):
        balance = starting_balance
        peak = starting_balance
        equity = [starting_balance]
        consecutive_losses = 0
        weekly_losses = 0
        halted = False
        actual_trades = 0

        for trade_number in range(trades_per_run):
            if halted:
                break

            r_multiple = rng.choice(observed_r)
            # Calibration phase: cap risk at 0.5%, as in Section 13.
            trade_risk = 0.005 if trade_number < calibration_trades else risk_percent
            balance *= 1.0 + r_multiple * trade_risk
            balance = max(balance, 0.0)
            actual_trades += 1

            peak = max(peak, balance)
            equity.append(balance)

            if r_multiple < 0:
                consecutive_losses += 1
                weekly_losses += 1
            else:
                consecutive_losses = 0

            if consecutive_losses >= 2:
                halted = True
            if weekly_losses >= 4:
                trade_risk *= 0.5
                weekly_losses = 0
            if (peak - balance) / peak > 0.08:
                halted = True

        outcomes.append(MonteCarloResult(
            final_balance=balance,
            peak_balance=peak,
            max_drawdown=max_drawdown(equity),
            minimum_balance=min(equity),
            ruin=balance <= starting_balance * ruin_threshold,
            halted=halted,
            total_trades=actual_trades,
        ))

    finals = sorted(result.final_balance for result in outcomes)
    drawdowns = sorted(result.max_drawdown for result in outcomes)

    def percentile(sorted_values: Sequence[float], percent: float) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]
        index = (len(sorted_values) - 1) * percent
        lower = int(index)
        upper = min(lower + 1, len(sorted_values) - 1)
        fraction = index - lower
        return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction

    profitable = sum(result.final_balance > starting_balance for result in outcomes)
    ruined = sum(result.ruin for result in outcomes)
    halted = sum(result.halted for result in outcomes)

    return {
        "simulations": simulations,
        "trades_per_run": trades_per_run,
        "starting_balance": starting_balance,
        "observed_trade_count": len(observed_r),
        "observed_expectancy_r": sum(observed_r) / len(observed_r),
        "profitable_probability": profitable / simulations,
        "ruin_probability": ruined / simulations,
        "halt_probability": halted / simulations,
        "ending_balance_percentiles": {
            "p05": round(percentile(finals, 0.05), 2),
            "p50": round(percentile(finals, 0.50), 2),
            "p95": round(percentile(finals, 0.95), 2),
        },
        "max_drawdown_percentiles": {
            "p05": round(percentile(drawdowns, 0.05), 4),
            "p50": round(percentile(drawdowns, 0.50), 4),
            "p95": round(percentile(drawdowns, 0.95), 4),
        },
    }


def read_observed_r(csv_path: Path) -> list[float]:
    import csv

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "exit_r" not in reader.fieldnames:
            raise ValueError("CSV must contain an exit_r column")
        return [float(row["exit_r"]) for row in reader if row.get("exit_r")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo test for Unified Trading System")
    parser.add_argument("csv", type=Path, help="CSV containing historical exit_r values")
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--trades", type=int, default=200)
    parser.add_argument("--balance", type=float, default=10_000.0)
    parser.add_argument("--risk", type=float, default=0.0075)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    observed = read_observed_r(args.csv)
    report = run_monte_carlo(
        observed,
        simulations=args.simulations,
        trades_per_run=args.trades,
        starting_balance=args.balance,
        risk_percent=args.risk,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
