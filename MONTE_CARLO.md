
## Monte Carlo

`exit_r` is the result of a closed trade in R units. Keep the values empirical;
do not generate them from targets as a guarantee of fills.

```bash
python monte_carlo.py example_monte_carlo.csv --simulations 10000 --trades 200 --balance 10000 --risk 0.0075 --seed 42
pytest -q
```

The bootstrap model randomly samples and reorders observed trades. It does not
create new R-multiples or model correlations. If the sample is tiny or biased,
the result remains uncertain, even with many simulations.
