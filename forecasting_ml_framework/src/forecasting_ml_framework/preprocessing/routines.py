#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The routine library: every concrete transform, re-exported under one module.

The registry resolves routines by short name from this module. Keeping one
aggregate module means the fallback chain in
:func:`~forecasting_ml_framework.utils.reflection.resolve_routine` has a single
first candidate, and a routine that is defined anywhere in the family is
reachable without touching a node.

Every routine here is authored against one of three shapes:

* :class:`~forecasting_ml_framework.core.routine.PreprocessRoutine` for
  transforms whose state is small and picklable in place;
* :class:`~forecasting_ml_framework.preprocessing.bases.SparkMlRoutine` for
  transforms whose fitted state is a Spark ML pipeline model;
* :class:`~forecasting_ml_framework.preprocessing.bases.UdfRoutine` for
  transforms that delegate their arithmetic to a pure helper.
"""

from forecasting_ml_framework.preprocessing.bases import (  # noqa: F401 - re-exported in __all__
  SparkMlRoutine,
  UdfRoutine,
)
from forecasting_ml_framework.preprocessing.categorical import (
  CategoricalEncoding,
  HighLabelBinning,
  OptimalBinning,
  TargetEncoding,
)
from forecasting_ml_framework.preprocessing.scalar import (
  CapOutliers,
  DomainNormalizations,
  Imputations,
  KNNImputation,
  Normalizations,
  QuantileDiscretizer,
  TreeImputation,
)
from forecasting_ml_framework.preprocessing.sequence import (
  ChangeRatio,
  CoefficientOfVariation,
  FixedLength,
  SequenceAverage,
  SequenceDateDelta,
  SequenceDelta,
  SequenceDomainNormalizations,
  SequenceFillNa,
  SequenceLast,
  SequenceMax,
  SequenceMin,
  SequenceNormalize,
  SequenceRecency,
  SequenceSum,
  Similarity,
  Slope,
)

#: Every routine in the library, keyed by its configuration registration key.
#: The test-suite asserts that this map and the module's public names agree, so
#: a routine cannot be added without being reachable.
AVAILABLE_ROUTINES: dict[str, type] = {
  'Imputations': Imputations,
  'KNNImputation': KNNImputation,
  'TreeImputation': TreeImputation,
  'CapOutliers': CapOutliers,
  'Normalizations': Normalizations,
  'QuantileDiscretizer': QuantileDiscretizer,
  'DomainNormalizations': DomainNormalizations,
  'CategoricalEncoding': CategoricalEncoding,
  'TargetEncoding': TargetEncoding,
  'HighLabelBinning': HighLabelBinning,
  'OptimalBinning': OptimalBinning,
  'FixedLength': FixedLength,
  'SequenceFillNa': SequenceFillNa,
  'SequenceNormalize': SequenceNormalize,
  'SequenceDomainNormalizations': SequenceDomainNormalizations,
  'SequenceAverage': SequenceAverage,
  'SequenceSum': SequenceSum,
  'SequenceMin': SequenceMin,
  'SequenceMax': SequenceMax,
  'SequenceLast': SequenceLast,
  'SequenceDelta': SequenceDelta,
  'SequenceDateDelta': SequenceDateDelta,
  'SequenceRecency': SequenceRecency,
  'ChangeRatio': ChangeRatio,
  'CoefficientOfVariation': CoefficientOfVariation,
  'Slope': Slope,
  'Similarity': Similarity,
}

__all__ = ['AVAILABLE_ROUTINES', 'SparkMlRoutine', 'UdfRoutine', *sorted(AVAILABLE_ROUTINES)]
