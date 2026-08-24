# WoE / IV Binning Optimizer

Compares binning strategies for credit-risk scorecards and reports the
Weight of Evidence and Information Value each one produces, so you can pick the
right one per feature instead of guessing.

Given a feature matrix and a binary target, it bins every feature several ways,
scores each result, and writes out the best binning rules per feature along with
a ranked comparison across strategies.

## Weight of Evidence and Information Value

These are the two numbers this project exists to produce, so it is worth being
precise about what they are.

The target is binary: a row is either **good** (repaid, `y = 0`) or **bad**
(defaulted, `y = 1`). Note the convention, because it decides the sign of
everything below: this codebase treats `1` as bad.

### Weight of Evidence (WoE)

WoE compares how common a group of the bads are against how common the goods
are, as a log ratio of two distributions:

$$
\operatorname{WoE}_i = \ln\!\left(\frac{\mathrm{dist\_goods}_i}{\mathrm{dist\_bads}_i}\right) = \ln\!\left(\frac{\dfrac{g_i}{G}}{\dfrac{b_i}{B}}\right)
$$

where, for bin $i$:

| Symbol | Meaning |
|---|---|
| $g_i$ | Number of goods in bin $i$ |
| $b_i$ | Number of bads in bin $i$ |
| $G$ | Total goods in the feature |
| $B$ | Total bads in the feature |

Read it as: *how many times more likely is a good than a bad to land in this
bin, compared with the base rate.* A WoE of 0 means the bin looks exactly like
the population. Positive means goods are over-represented, negative means bads
are.

The intuition is the odds ratio from a $2 \times 2$ table: $e^{\operatorname{WoE}_i}$ is
the odds ratio of bin $i$ against the population base rate. Rearranging the
definition gives the relationship that makes WoE usable as a predictor, namely
that a bin's log-odds is an affine function of its WoE:

$$
\ln\!\left(\frac{b_i}{g_i}\right) = \underbrace{\ln\!\left(\frac{b_i}{B}\right) - \ln\!\left(\frac{g_i}{G}\right)}_{\textstyle -\,\operatorname{WoE}_i} + \ln\!\left(\frac{B}{G}\right)
$$

The final term is the population log-odds and does not depend on the bin, so WoE
is the bin's log-odds expressed relative to the base rate. That is why WoE lines
up with the linear predictor in a logistic regression, and why scorecards are
usually built as a sum of WoE terms.

Note carefully that WoE is *not* additive across bins in the way a sum of odds
ratios would be: $e^{\sum_i \operatorname{WoE}_i}$ is the **product** of the
per-bin odds ratios, which is not the odds ratio of the pooled group. To pool
bins, aggregate the counts first and then apply the definition once.

The sign also gives you the direction of risk in one glance:

| WoE | Reading |
|---|---|
| $\operatorname{WoE} > 0$ | Goods over-represented, so the bin is *lower* risk |
| $\operatorname{WoE} = 0$ | No signal, the bin looks like the population |
| $\operatorname{WoE} < 0$ | Bads over-represented, so the bin is *higher* risk |

Read the sign against your own target definition, not against intuition. If bad
means "high value of this feature", expect negative WoE as the feature rises; if
bad means "low value of this feature", expect positive. The sign is set entirely
by how you encoded the target, so a feature can go either way in two different
projects without anything about the feature changing.

### Information Value (IV)

IV is the single number summarising how much predictive power a feature
carries. It is built from WoE, one contribution per bin:

$$
\operatorname{IV} = \sum_{i=1}^{k} \left(\mathrm{dist\_goods}_i - \mathrm{dist\_bads}_i\right) \cdot \ln\!\left(\frac{\mathrm{dist\_goods}_i}{\mathrm{dist\_bads}_i}\right) = \sum_{i=1}^{k} \underbrace{\left(\mathrm{dist\_goods}_i - \mathrm{dist\_bads}_i\right)\!\cdot \operatorname{WoE}_i}_{\text{IV contributed by bin } i}
$$

Equivalently, IV is the symmetrised Kullback-Leibler divergence between the good
and bad distributions:

$$
\operatorname{IV} = D_{\mathrm{KL}}\!\left(P_{\mathrm{goods}} \,\|\, P_{\mathrm{bads}}\right) + D_{\mathrm{KL}}\!\left(P_{\mathrm{bads}} \,\|\, P_{\mathrm{goods}}\right)
$$

It is zero when a feature carries no information, and it grows with how well the
feature separates the two classes. It is **not** bounded, which is why IV above
1 is arithmetically fine even though it looks alarming.

The widely used interpretation bands, for the usual "predicts default" case:

| IV | Strength | Common response |
|---|---|---|
| $\operatorname{IV} < 0.02$ | Not predictive | Drop |
| $0.02 \le \operatorname{IV} < 0.1$ | Weak | Keep, expect little lift |
| $0.1 \le \operatorname{IV} < 0.3$ | Medium | Useful |
| $0.3 \le \operatorname{IV} < 0.5$ | Strong | Good predictor |
| $\operatorname{IV} \ge 0.5$ | Suspicious | Almost certainly overfitted |

That last row is a warning, not a compliment. Very high IV usually means the
bins were chosen using the same rows being scored, so the number is partly
describing noise it memorised. A feature can also reach a high IV for the wrong
reason, for instance when a bin holds only a handful of rows whose bad rate
happens to be extreme, which is the failure mode `min_bin_size` exists to
prevent.

The other structural expectation is that **WoE should be monotone in the
feature value**. If a feature's risk genuinely rises with the value, its WoE
should trend one way without zig-zagging. A jagged trend means either noise in
small bins or a non-monotone relationship that the binning is failing to
capture, and both are worth knowing about. The `monotonic` column in the
comparison report flags exactly this.

### What the numbers cannot tell you

IV is computed on the same data that chose the bins, so it is optimistically
biased in proportion to how hard the search worked. That bias is why
`iv_optimized`, `optimal_dp` and `greedy_iv` score higher than `equal_freq` on
almost every feature in the reports below: they searched harder for the split
that suits these rows. Use them to pick a *sensible* binning, then re-score it
on a held-out split before believing the IV.

The search helpers in `binning/base.py` apply additive smoothing so that a bin
with no goods or no bads yields a finite WoE instead of an infinity:

$$
\mathrm{dist\_goods}_i = \frac{g_i + \alpha}{G + \alpha}, \qquad \mathrm{dist\_bads}_i = \frac{b_i + \alpha}{B + \alpha}, \qquad \alpha = 0.5
$$

`woe_iv_calculator.py`, which produces the reported figures, does not smooth and
instead maps infinities to zero. See Known issues.

## Quick start

```bash
pip install -r requirements.txt
cd woe_iv_optimizer
python src/main.py
```

Run commands from the `woe_iv_optimizer` directory, since `main.py` resolves
`config/config.yaml` and all output paths relative to the working directory.
The shipped config is a template pointing at `data/input.csv`, so point
`data_source` and the file path at your own data first.

`main.py` reads `config/config.yaml`. To compare every strategy on a dataset and
produce the ranking report:

```bash
python src/compare_strategies.py
```

Output paths are derived from the config file name, so pointing `--config` at a
second config writes to `logs/strategy_comparison_<name>.log` and
`output/strategy_comparison_<name>.csv` instead of overwriting the first.

## Configuration

`config/config.yaml` drives everything. The parts that matter most:

```yaml
data_source: "sqlite"          # bigquery | csv | parquet | sqlite
target_column: "is_default"    # 1 = bad (default), 0 = good
score_column: ""               # excluded from the feature list if set
features: []                   # empty = auto-detect every non-target column

binning_strategies:            # any subset of the names listed below
  - "equal_freq"
  - "optimal_dp"

constraints:
  min_bin_size: 0.05           # minimum share of rows per bin
  max_bins: 10                 # hard cap on bin count
  enforce_monotonicity: false
  min_iv_threshold: 0.02       # below this, no plot is written

missing_handling:
  strategy: "separate_bin"     # separate_bin | impute_median | drop
```

`enforce_monotonicity` only affects the strategies that ask for it. PAVA and
Siddiqi honour it natively; the IV-greedy and DP strategies treat it as a
post-hoc repair, and the purely unsupervised ones ignore it, since monotonic
WoE is meaningless without a target.

## Data sources

| `data_source` | Notes |
|---|---|
| `csv` | Local path, or `gs://` / `s3://` / any URL pandas can read |
| `parquet` | Local file |
| `sqlite` | Needs `sqlite.file_path` plus either `sqlite.table` or `sqlite.query` |
| `bigquery` | Optional dependency, see Known issues |

The SQLite loader takes either a table name or a full query, and the query wins
when both are set:

```yaml
sqlite:
  file_path: "data/input.db"
  table: "my_table"
  # query: "SELECT * FROM my_table WHERE is_default IS NOT NULL"
```

`features` is auto-detected when left empty, which means string columns end up in
the feature list. The 15 numeric strategies cannot cut on unordered values, so
`strategies_for_feature` routes any non-numeric feature to `categorical` instead
of failing 15 times per column.

## Binning strategies

All 16 are registered in `src/binning_factory.py` and selectable by name.

| Name | Approach | Notes |
|---|---|---|
| `equal_width` | Equal-width intervals | Degrades on skewed features |
| `equal_freq` | Quantile cuts | Safe default |
| `winsorized` | Cap at 1%/99%, then quantile | Fixes heavy-tailed features |
| `natural_breaks` | Fisher-Jenks | Unsupervised, good as a control |
| `kmeans` | 1-D Lloyd | Unsupervised, population-free |
| `tree_based` | Decision tree leaves | Fast, strong on tabular data |
| `cart_iv` | Tree pruned by IV loss | Prunes on IV, not impurity |
| `chimerge` | Chi-square merge | `chi_threshold` controls significance |
| `siddiqi` | Greedy min-IV-loss merge | Scorecard standard |
| `greedy_iv` | Forward selection | Stops when a cut stops paying |
| `iv_optimized` | Grid search on IV | Slow, see below |
| `optimal_dp` | Exact IV max via DP | Optimal over the provisional grid |
| `monotonic_pava` | PAVA fit | Tries both WoE directions |
| `mdl` | Min description length | Penalises fine binning |
| `bic` | Min BIC | Strongest regularisation |
| `categorical` | Rare bucket + IV merge | Numeric input falls back to `equal_freq` |

The DP, MDL and BIC strategies solve a shortest-path problem over a provisional
quantile grid rather than searching all possible binning. That makes them exact
*given the grid*, not exact over every partition.

`iv_optimized` is the one to watch. It cannot enumerate all binnings of a
high-cardinality feature. With 80 distinct values and `max_bins: 10` that is
about $2 \times 10^{11}$ candidate binnings, so it searches a candidate grid
sized `max_bins + 2` instead. It takes far longer per feature than every other
strategy here and dominates the runtime of a full comparison. Use `optimal_dp` if
you want the same objective far faster.

## Reading the results

`logs/strategy_comparison.log` has four sections: the best strategy per feature,
a full IV pivot, per-strategy aggregates (mean IV, median IV, runtime, failure
count), and any failures. The matching CSV has one row per feature and strategy
with `iv`, `bins`, `min_bin_share`, `monotonic`, `seconds` and `ok`.

Rank the strategies on your own data rather than assuming a winner. The usual
pattern is that `tree_based` and `optimal_dp` lead on tabular data, `mdl` and
`bic` trade some IV for far fewer bins, and `monotonic_pava` and `greedy_iv`
score lowest by design, since PAVA spends IV to buy monotonicity and greedy
stops as soon as a cut stops paying. Treat the mean IV as relative, not
absolute: it moves with your target definition, your `max_bins`, and how hard the
search worked.

## Layout

```
config/          config.yaml
src/
  main.py              pipeline entry point
  compare_strategies.py  strategy comparison and report
  binning_factory.py   strategy name -> class registry
  binning/
    base.py            shared search primitives and helpers
    <strategy>.py      one file per strategy
  data_loader/         bigquery, csv, parquet, sqlite
  woe_iv_calculator.py WoE and IV
  monotonicity_checker.py
  visualizer.py
tests/
data/, output/, logs/  generated, not checked in
```

## Conventions

2-space indentation, single quotes, Google-style docstrings. The combinatorial
strategies share helpers in `binning/base.py` (`prebin`, `optimal_partition`,
`single_bin`), so add new strategies there rather than duplicating the search.

## Known issues

- `bigquery` is broken. `BigQueryDataLoader.__init__` reads
  `config['bigquery']['external_project']`, but the config only defines
  `project_id`, `dataset_id` and `table_id`. The import is optional so the other
  three sources work without `google-cloud-bigquery` installed, but selecting
  `data_source: "bigquery"` will raise a `KeyError`.
- Only 2 unit tests exist, both covering `equal_freq` and `iv_optimized`. The 11
  newer strategies have no automated tests, so they are verified only by running
  the end-to-end comparison on your own data.
- `pylint` is clean at 10.00/10 against the repository `.pylintrc`. There is no
  `pyproject.toml` or ruff config, so `ruff` has to be run with `--isolated` and
  will otherwise report every single-quoted string as a style error.
  `requirements.txt` also still lists `seaborn`, which is no longer imported.
- `woe_iv_calculator.py` and the search helpers in `binning/base.py` do not
  agree on smoothing. The helpers use additive 0.5 smoothing, while the
  calculator divides the raw distributions and maps infinities to zero. So the
  IV a strategy optimises for is not bit-for-bit the IV that gets reported, and
  they diverge most on bins with no goods or no bads. Worth unifying.
- Monotonicity is reported per binning but never enforced across strategies, and
  IV is computed on the same data used to choose the bins, so the reported IV is
  optimistically biased. Use an out-of-sample split before trusting it.
