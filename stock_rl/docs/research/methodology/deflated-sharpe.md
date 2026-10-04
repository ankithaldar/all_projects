# Deflated Sharpe Ratio — Verification and Fixes

Our shipped `deflated_sharpe` was **materially wrong in three places**.
Verified against Bailey & Lopez de Prado, *"The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality"*,
Journal of Portfolio Management, 2014, plus *Advances in Financial
Machine Learning* (2018).

## The published formulas

```
SR0 = sqrt(V[{SR_n}]) · [ (1-γ)·Z⁻¹[1 - 1/N] + γ·Z⁻¹[1 - 1/(N·e)] ]     (1)

                (SR̂ - SR0)·sqrt(T-1)
DSR = Z [ ─────────────────────────────────── ]                          (2)
              [ sqrt(1 - γ₃·SR̂ + ((γ₄-1)/4)·SR̂²) ]
```

- `V[{SR_n}]` is **the variance across the trials' estimated Sharpe**
- `γ` = Euler–Mascheroni, **appears only in SR0**
- `γ₃` = skewness, `γ₄` = **raw** kurtosis (3 for Gaussian). Different
  symbols entirely from `γ`.

## The three bugs

### 1. `sqrt(T-1)` counted years, not observations

`T` is the number of **observations**, and `SR̂` is the **non-annualised**
Sharpe. AFML §14.7.2 states this explicitly.

Because we annualised the Sharpe consistently, the correct annualised-units
factor is `sqrt((T-1)/P)`. We used `sqrt(T/P - 1)`:

| T (daily) | years | correct `√((T−1)/P)` | ours `√(years−1)` | error |
|---|---|---|---|---|
| 1250 | 5.0 | 2.235 | 2.000 | −10% |
| 504 | 2.0 | 1.413 | 1.000 | −29% |
| 252 | 1.0 | 0.998 | 0.000 | **−100%** |

The error grows as the sample shortens — the opposite of where accuracy is
needed.

### 2. The denominator used Euler–Mascheroni instead of skewness and kurtosis

Correct: `1 − γ₃·SR̂ + ((γ₄−1)/4)·SR̂²`. We had
`1 − γ·observed + γ·expected_max²`.

Consequences: the SR² coefficient should be `kurtosis/4 ≈ 2.5` for fat
tails, not `0.577`. And **it goes negative**:

```
annualised SR 1.73 -> spread +0.0014
annualised SR 2.00 -> spread -0.1544  => return 0.0
annualised SR 2.50 -> spread -0.4430  => return 0.0
```

**Our code reported DSR = 0.0 for any strategy above ~1.7 annualised
Sharpe — annihilating exactly the candidates it should endorse.**

### 3. Trial dispersion was the strategy's own volatility

Eq. (1) uses `sqrt(V[{SR_n}])`, the dispersion of trial Sharpes. The
paper's worked example makes it concrete: `sqrt(1/2)` divided by `sqrt(250)`
to put trial-Sharpe dispersion in the same per-period units as the
observed Sharpe. It has **nothing to do with the strategy's return
volatility**.

Magnitude at N=100:

| Scale used | SR0 (annualised) |
|---|---|
| our proxy, vol=0.10 | 0.253 |
| our proxy, vol=0.20 | 0.506 |
| **correct, σ = 1/√5 (unskilled trials)** | **1.132** |
| correct, paper's V[SR]=0.5 | 1.789 |

We understated the bar by **2–7x**, so systematically over-reported
significance — and the error was *inversely* related to the strategy's
volatility, penalising high-vol strategies for being high-vol.

## The fix, verified against the paper

Paper's worked example (p.10): N=100, `V[SR]=1/2`, T=1250, annualised
SR=2.5, γ₃=−3, γ₄=10, 250 obs/yr.

| Quantity | Paper | Our reconstruction |
|---|---|---|
| max_z(100) | — | 2.530603 |
| SR0 | 0.1132 | 0.113172 |
| DSR (N=100) | 0.9004 | **0.900397** |
| DSR (N=46) | 0.9505 | **0.950502** |
| DSR (N=88, normal) | 0.9505 | **0.950491** |

All four reproduce. These are pinned as regression tests in
`tests/test_metrics.py`.

```python
def dsr_from_moments(sharpe, expected_max_sharpe, skew, kurtosis, observations):
  variance = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe ** 2
  if variance <= 0.0:
    return 0.0
  numerator = (sharpe - expected_max_sharpe) * math.sqrt(observations - 1)
  return _NORMAL.cdf(numerator / math.sqrt(variance))
```

`dsr_from_moments` is separated from `deflated_sharpe` so the paper's
example can be reproduced with the skewness and kurtosis it *states*
rather than values estimated from a sample.

## Two behavioural changes

**Undefined is no longer 0.0.** `deflated_sharpe` returns `None` when the
series has no dispersion or fewer than two observations. Zero asserts
"not credible after search", which is a **substantive claim**; an
insufficient sample supports no claim at all.

**The `years <= 1.0` guard was wrong.** `T > 1` means observations, not
years. A 200-day backtest is a perfectly valid sample. That guard was
discarding all sub-year data and, per the `sqrt` error above, was most
wrong exactly in the 1–2 year range.

## Minimum Backtest Length — run this as a gate

Bailey, Borwein, Lopez de Prado & Zhu (2014), *Notices of the AMS* 61(5):

```
MinBTL(years) = ( max_z(N) / target_annualised_SR )²
```

Reproduces their headline claims: N=7 → 1.92 yrs ("2 years → no more than
7 trials"); N=45 → 5.00 yrs. Their loose upper bound is `2·ln(N)/SR²`.

**Operational rule: before trusting a DSR, check MinBTL ≤ your sample
length.** N=100 needs 10.6 years; N=1000 needs 47. If you do not have
that, DSR ≈ 0 is the honest answer and no implementation correctness helps.

## Is CPCV necessary?

**No, not yet.** The DSR needs two inputs — N and cross-trial Sharpe
dispersion — and both come from **logging your parameter sweep**, not from
cross-validation. Purged walk-forward is sufficient for a first correct
implementation.

Two caveats from AFML §11.6: walk-forward has a single path that "can be
repeated over and over until a false positive appears", and it needs
serial dependence handled, hence embargo.

Add CPCV (AFML §12.4, φ[N,k] = C(N,k)·∏(N−i)/(k−1)!) when you want the
empirical Sharpe *distribution* or PBO.

## Embargo sizing

AFML §7.4.2: *"A small value h ≈ .01T often suffices to prevent all
leakage"* — **~1% of the sample**, implemented as
`mbrg = int(X.shape[0] * pctEmbargo)` bars. The chapter exercise uses
10-fold with 1% embargo. No fixed "correct" percentage; pick it by
confirming performance stops improving as you inflate it.

Direction matters: embargo applies **only to training observations after
the test set**, never before.

## Limitations — state these, do not hide them

**DSR is not a calibrated test.** `SR0` is the *mean* of the null
distribution of the maximum, not its (1−α) quantile. At `SR = SR0` the
numerator is exactly zero, so **DSR = 0.5 by construction**. The 0.95 bar
is a convention layered on top, not an α-level.

Simulation of best-of-N searches on **pure zero-edge Gaussian returns**,
T=1260 (5y), σ = 1/√years:

| N | E[max SR_ann] | P(naive PSR > 0.95) | **P(DSR > 0.95)** |
|---|---|---|---|
| 5 | 0.52 | 0.223 | **0.021** |
| 20 | 0.84 | 0.637 | **0.087** |
| 100 | 1.14 | 0.990 | **0.360** |
| 500 | 1.35 | 1.000 | **0.843** |

A massive improvement over the naive test, but at N=100 it fires on **36%
of pure-noise searches**.

Other real limitations:

- **Inputs are unobservable.** N is never reported; `V[{SR_n}]` requires
  logging everything. Undercounting N weakens deflation silently.
- The denominator is a local asymptotic expansion. Accurate to 0.6% at
  realistic daily Sharpe with Gaussian returns, but it **over-estimates
  true variance by 18–29x** at per-period Sharpe ≈ 1 with kurtosis ≈ 95.
- **No autocorrelation adjustment.** Lo (2002): serial correlation
  materially changes the Sharpe estimator's standard error, with annualised
  Sharpes overstated by up to 65%.
- **Goodhart's law applies to DSR itself.** *"when a measure becomes a
  target, it ceases to be a good measure"* (PBO paper §5.2).
- **The false strategy theorem** (Bailey & LdP, *Significance* 2021,
  18(6)): with enough trials, E[max SR] is right-unbounded — *"as few as
  three independent trials suffice to produce an investment strategy that
  is likely false"*.

## Which multiple-testing correction to use

| Method | Controls | Needs full trial P&L matrix? | Answers |
|---|---|---|---|
| **DSR** | multiplicity + non-normality | No — needs only N and dispersion | "Is this Sharpe special given how hard I searched?" |
| **White's Reality Check** | FWER | Yes | "Does the best model beat the benchmark after snooping?" |
| **Hansen SPA** | FWER, studentized | Yes | Same, robust to irrelevant alternatives — **prefer over White RC** |
| **Benjamini–Hochberg FDR** + Harvey–Liu | FDR | No (just p-values) | "Which of my many factors survive screening?" |
| **PBO / CSCV** | IS→OOS rank correlation | Yes | "Does my selection process work at all?" |

For a single strategy with a known N, **DSR is the right primary tool**.
Pair it with **Hansen SPA** (not White's RC) if you have logged the full
sweep — they answer different questions. Add **PBO** if you want to know
whether the pipeline generates signal or noise. Run **MinBTL first**.

**Harvey, Liu & Zhu (2016, RFS 29(1))**: a new factor needs **t > 3.0**,
not 2.0. Of 296 published significant factors, **158 are false
discoveries** under their framework.