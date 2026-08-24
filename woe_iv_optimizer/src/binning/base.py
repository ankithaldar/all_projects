#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Abstract Class for Binning Strategies"""


# imports
from abc import ABC, abstractmethod
import math
import pandas as pd
import numpy as np
from typing import Callable, List, Tuple
#    script imports
from monotonicity_checker import MonotonicityChecker
# imports


# constants
# constants


# functions
def prebin(x: pd.Series, n_bins: int) -> tuple:
  """Reduce a feature to a small set of provisional bins.

  Every combinatorial strategy in this package searches over groupings of a
  small provisional grid rather than over the raw distinct values, which is
  what keeps the searches tractable. Equal-frequency quantile cuts are used
  as that grid because they respect the population floor by construction and
  give every downstream merge a balanced starting point.

  Args:
    X: Numeric feature, possibly containing missing values.
    n_bins: Requested number of provisional bins.

  Returns:
    tuple: ``(codes, edges)`` where ``codes`` is a Series of provisional bin
    positions aligned to ``X`` and ``edges`` is the ascending list of cut
    points, or ``(None, [])`` when the feature is too sparse to bin.
  """
  clean = x.dropna()
  n_bins = min(n_bins, clean.nunique())
  if len(clean) == 0 or n_bins < 2:
    return None, []

  try:
    codes, edges = pd.qcut(clean, q=n_bins, retbins=True, duplicates='drop')
  except ValueError:
    return None, []

  if len(edges) < 3:
    return None, []

  # Position of each observation among the provisional bins, which is what the
  # combinatorial searches manipulate. -1 marks a missing value.
  positions = pd.Series(codes.cat.codes, index=clean.index)
  return positions, [float(e) for e in edges]


def is_numeric_feature(x: pd.Series) -> bool:
  """Whether a feature can be binned by the numeric strategies.

  Args:
    X: Feature to inspect.

  Returns:
    bool: True when the dtype is numeric.
  """
  return bool(pd.api.types.is_numeric_dtype(x))


def strategies_for_feature(x: pd.Series, configured: list) -> list:
  """Resolve which strategies to run for a given feature.

  Every strategy except ``categorical`` cuts on ordered numeric values, so a
  string column makes all of them fail. ``main.py`` auto-detects features, so a
  single categorical column in an otherwise numeric dataset would otherwise be
  skipped with an error per strategy. Route those to ``categorical`` instead.

  Args:
    X: Feature to be binned.
    configured: Strategy names requested by the config.

  Returns:
    list: Strategy names that can actually handle this feature.
  """
  if is_numeric_feature(x):
    return list(configured)
  return ['categorical']


def single_bin(x: pd.Series, label: str = 'All') -> tuple:
  """Collapse a feature into one bin.

  A search that concludes the feature is not worth splitting must still produce
  a single bin. Returning the untouched values instead would leave every
  distinct value looking like its own bin, which reads as a constraint
  violation and produces a meaningless IV.

  Args:
    X: Numeric feature.
    label: Label to use for the resulting bin.

  Returns:
    tuple: A series filled with ``label`` and a single-interval edge list.
  """
  return pd.Series(label, index=x.index, dtype='object'), [-np.inf, np.inf]


def edges_for_boundaries(prebin_edges: list, boundaries: list) -> list:
  """Translate provisional bin boundaries into feature-value cut points.

  The combinatorial strategies work on provisional bin *positions*, which are
  indices, not feature values. Converting a position boundary into a numeric
  cut point has to go through the pre-binning edges, otherwise the resulting
  edges land on an arbitrary numeric scale and every observation collapses
  into a single bin.

  Args:
    prebin_edges: The ``m + 1`` edges returned by :func:`prebin`, where
      ``prebin_edges[p]`` is the left edge of provisional position ``p``.
    boundaries: Provisional positions at which a new bin should start.

  Returns:
    list: Ascending interior edges with infinite outer bounds, or an empty list
    when the boundaries do not describe a real split.
  """
  if len(prebin_edges) < 3:
    return []

  interior = []
  for boundary in sorted(set(int(b) for b in boundaries)):
    if 0 < boundary < len(prebin_edges) - 1:
      value = float(prebin_edges[boundary])
      if not interior or value > interior[-1]:
        interior.append(value)
  return interior


def full_edges(interior: list) -> list:
  """Wrap interior cut points with infinite outer bounds.

  Args:
    interior: Ascending interior edges.

  Returns:
    list: ``[-inf, *interior, inf]``, suitable for :func:`pandas.cut`.
  """
  return [-np.inf] + list(interior) + [np.inf]


def monotonic_repair(edges: list, x: pd.Series, y: pd.Series) -> list:
  """Merge adjacent bins until the WoE trend becomes monotone.

  Some strategies optimise a criterion that has nothing to do with shape, so
  their output can zig-zag. Rather than discarding such a result, this repairs
  it by pooling adjacent bins, which is the same idea PAVA applies but applied
  after the fact.

  Args:
    edges: Current bin edges, as returned by :func:`full_edges`.
    X: Numeric feature.
    y: Binary target.

  Returns:
    list: Edges with non-monotone bins merged. Returned unchanged when the
    input is already monotone or cannot be repaired.
  """
  interior = [e for e in edges if np.isfinite(e)]
  if len(interior) < 1:
    return edges

  binned = pd.cut(x, bins=edges)
  woe_by_bin = woe_per_bin(binned, y)
  if woe_by_bin is None or MonotonicityChecker.is_monotonic(woe_by_bin):
    return edges

  # Pool adjacent bin pairs for as long as the monotone fit is violated.
  # ``edges`` is [-inf, e0, ..., ek, inf] and bin i spans
  # (edges[i], edges[i+1]], so a block starting at bin b0 keeps its left cut
  # at edges[b0].
  values = list(woe_by_bin)
  blocks: List[int] = list(range(len(values)))
  while len(blocks) > 1:
    violations = [i for i in range(len(blocks) - 1)
                  if values[blocks[i]] > values[blocks[i + 1]]]
    if not violations:
      break
    del blocks[violations[0] + 1]

  interior = [edges[b0] for b0 in blocks[1:]]
  return full_edges(interior)


def woe_per_bin(binned: pd.Series, y: pd.Series) -> list | None:
  """WoE of each bin, ordered by ascending bin.

  Args:
    binned: Binned feature.
    y: Binary target.

  Returns:
    list | None: WoE per ascending bin, or None when the target is
    single-class and WoE is undefined.
  """
  frame = pd.DataFrame({'bin': binned, 'y': y})
  agg = frame.groupby(
    'bin', observed=True, dropna=False)['y'].agg(['count', 'sum'])
  if agg.empty:
    return None
  total = agg['count'].to_numpy(dtype=float)
  bads = agg['sum'].to_numpy(dtype=float)
  goods = total - bads
  if goods.sum() <= 0 or bads.sum() <= 0:
    return None
  return list(woe_from_counts(goods, bads, goods.sum(), bads.sum()))


def counts_by_position(positions: pd.Series, y: pd.Series) -> tuple:
  """Tabulate goods and bads per provisional bin position.

  Args:
    positions: Provisional bin positions from :func:`prebin`.
    y: Binary target aligned to the same index as ``positions``.

  Returns:
    tuple: ``(goods, bads)`` integer arrays, one entry per position present in
    ``positions``, and the number of positions.
  """
  frame = pd.DataFrame({
    'pos': positions.to_numpy(),
    'y': y.loc[positions.index].to_numpy(),
  })
  agg = frame.groupby('pos', observed=True)['y'].agg(['count', 'sum'])
  n_positions = int(positions.max()) + 1 if len(positions) else 0

  bads = np.zeros(n_positions, dtype=float)
  total = np.zeros(n_positions, dtype=float)
  for pos, row in agg.iterrows():
    bads[pos] = row['sum']
    total[pos] = row['count']
  return total - bads, bads


def woe_from_counts(goods: np.ndarray, bads: np.ndarray,
                    total_goods: float, total_bads: float) -> np.ndarray:
  """Weight of Evidence for arbitrary group counts.

  Applies the additive 0.5 smoothing used throughout credit-risk practice so
  that empty goods or bads inside a group do not produce infinite WoE.

  Args:
    goods: Number of good observations per group.
    bads: Number of bad observations per group.
    total_goods: Good observations across the whole feature.
    total_bads: Bad observations across the whole feature.

  Returns:
    np.ndarray: WoE per group.
  """
  if total_goods <= 0 or total_bads <= 0:
    return np.zeros(len(goods), dtype=float)
  dist_goods = (goods + 0.5) / (total_goods + 0.5)
  dist_bads = (bads + 0.5) / (total_bads + 0.5)
  return np.log(dist_goods / dist_bads)


def iv_from_counts(goods: np.ndarray, bads: np.ndarray,
                   total_goods: float, total_bads: float) -> np.ndarray:
  """Per-group Information Value contribution.

  Args:
    goods: Number of good observations per group.
    bads: Number of bad observations per group.
    total_goods: Good observations across the whole feature.
    total_bads: Bad observations across the whole feature.

  Returns:
    np.ndarray: IV contribution per group.
  """
  if total_goods <= 0 or total_bads <= 0:
    return np.zeros(len(goods), dtype=float)
  dist_goods = (goods + 0.5) / (total_goods + 0.5)
  dist_bads = (bads + 0.5) / (total_bads + 0.5)
  woe = np.log(dist_goods / dist_bads)
  return (dist_goods - dist_bads) * woe


def prefix_sums(values: np.ndarray) -> np.ndarray:
  """Cumulative sums with a leading zero, for O(1) range queries.

  Args:
    values: Numeric array.

  Returns:
    np.ndarray: Array of length ``len(values) + 1`` where entry ``i`` is the
    sum of ``values[:i]``.
  """
  return np.concatenate([[0.0], np.cumsum(values)])


def optimal_partition(cost: Callable[[int, int], float], n_items: int,
                      min_items: int, max_parts: int) -> list:
  """Partition a sequence into the minimum-cost contiguous groups.

  This is the shared search behind the dynamic-programming strategies. Because
  the items are already in a meaningful order (usually ascending feature
  value), any valid binning corresponds to a contiguous partition, so the
  problem reduces to a shortest-path problem solved exactly in O(n^2 * k).

  Args:
    cost: Callable returning the cost of grouping items ``[i, j)``. Must be
      additive across groups for the result to be a true optimum.
    n_items: Number of provisional items to partition.
    min_items: Minimum number of items per group.
    max_parts: Maximum number of groups.

  Returns:
    list: List of ``(start, stop)`` half-open index pairs covering
    ``[0, n_items)`` contiguously, or an empty list when no feasible partition
    exists.
  """
  if n_items <= 0:
    return []

  # feasible[j][k] is (lowest cost, split point) for covering the first j
  # items with k groups, or None when that is not reachable.
  feasible = [[None] * (max_parts + 1) for _ in range(n_items + 1)]
  feasible[0][0] = (0.0, None)

  for j in range(1, n_items + 1):
    for k in range(1, min(max_parts, j // max(1, min_items)) + 1):
      best_cost = math.inf
      best_split = -1
      for i in range(j - min_items + 1):
        prev = feasible[i][k - 1]
        if prev is None:
          continue
        candidate = prev[0] + cost(i, j)
        if candidate < best_cost:
          best_cost, best_split = candidate, i
      if best_split >= 0:
        feasible[j][k] = (best_cost, best_split)

  overall_cost = math.inf
  overall_parts = 0
  for k in range(1, max_parts + 1):
    entry = feasible[n_items][k]
    if entry is not None and entry[0] < overall_cost:
      overall_cost, overall_parts = entry[0], k

  if not overall_parts:
    return []

  groups = []
  j, k = n_items, overall_parts
  while k > 0:
    i = feasible[j][k][1]
    groups.append((i, j))
    j, k = i, k - 1
  groups.reverse()
  return groups


def groups_to_edges(groups: list, n_items: int) -> list:
  """Map provisional group indices back onto original feature edges.

  Args:
    groups: Contiguous ``(start, stop)`` pairs over provisional positions.
    n_items: Number of provisional items, used to validate the mapping.

  Returns:
    list: Ascending bin edges, or an empty list when ``groups`` does not cover
    every provisional item exactly once.
  """
  if not groups:
    return []
  covered = 0
  for start, stop in groups:
    if start != covered or stop <= start:
      return []
    covered = stop
  if covered != n_items:
    return []
  return list(groups)
# functions


# classes
class BinningStrategy(ABC):
  """Abstract Class for Binning Strategies"""

  def __init__(self, min_bin_size: float = 0.05, max_bins: int = 10,
               enforce_monotonicity: bool = False,
               **kwargs):  # pylint: disable=unused-argument
    # kwargs absorbs strategy-specific options such as chi_threshold, which the
    # factory forwards to every strategy but only some of them accept.
    self.min_bin_size = min_bin_size
    self.max_bins = max_bins
    self.enforce_monotonicity = enforce_monotonicity

  @abstractmethod
  def bin(self, x: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    """Return binned series and bin edges (or categories)."""
    pass

# classes
