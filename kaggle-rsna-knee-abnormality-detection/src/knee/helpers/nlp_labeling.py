#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rule-based pseudo-labeling from radiology reports (MVP).

Strategy per target:

1. Simple targets match an affirmative lexicon; each mention is classified
   positive / negated / uncertain by scanning a token window before it.
2. Compartment OA targets require co-occurrence of an OA anchor term and a
   compartment locator inside a token window (either side may be negated).

Verdict aggregation across mentions: any affirmed mention wins, else any
uncertain mention yields ``UNKNOWN`` (``-1``), else any negated mention
yields 0, otherwise the target was never mentioned and stays ``UNKNOWN``.
The ``-1`` sentinel is consumed downstream as ``ignore_index`` by losses so
uncertain or unmentioned targets never contribute gradients.

Multilingual reports are out of scope here (backlog: XLM-R labeler); the
notebook 02 language census sizes that follow-up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

POSITIVE = 1
NEGATED = 0
UNKNOWN = -1

DEFAULT_NEGATION_TRIGGERS = (
    'no',
    'not',
    'without',
    'intact',
    'unremarkable',
    'normal',
    'negative',
    'absent',
    'free',
    'resolved',
    'unchanged',
)

DEFAULT_UNCERTAIN_TRIGGERS = (
    'rule',
    'out',
    'cannot',
    'exclude',
    'excluded',
    'question',
    'questionable',
    'possible',
    'r/o',
)

_OA_ANCHOR = (
    r'\bosteoarthrit\w*\b'
    r'|\bosteoarthros\w*\b'
    r'|\bdegenerative\s+(?:change|changes|disease|joint\s+disease)\w*\b'
    r'|\bjoint\s+space\s+(?:narrowing|loss)\b'
    r'|\bcartilage\s+(?:loss|thinning|wear)\b'
    r'|\bchondromalaci\w*\b'
    r'|\boa\b'
)


@dataclass(frozen=True)
class TargetSpec:
    """Declarative matching rule for one abnormality target.

    Attributes:
        name: Column name in train.csv / submission header.
        lexicon: Affirmative pattern; required when ``locator`` is None.
        locator: Co-occurrence partner pattern for compartment targets.
        cooccurrence_window: Token radius between anchor and locator.
    """

    name: str
    lexicon: str | None = None
    locator: str | None = None
    cooccurrence_window: int = 8


TARGET_SPECS: tuple[TargetSpec, ...] = (
    TargetSpec(name='ACL', lexicon=r'\bacl\b|\banterior\s+cruciate(?:\s+ligament)?\b'),
    TargetSpec(name='MCL', lexicon=r'\bmcl\b|\bmedial\s+collateral(?:\s+ligament)?\b'),
    TargetSpec(
        name='Medial Meniscus',
        lexicon=(
            r'\bmedial\s+menisc(?:us|i)\b'
            r'|\bmedial\s+meniscal\b'
            r'|\bmmedial\s+menisc\w*\b'
        ),
    ),
    TargetSpec(
        name='Lateral Meniscus',
        lexicon=r'\blateral\s+menisc(?:us|i)\b|\blateral\s+meniscal\b',
    ),
    TargetSpec(
        name='Medial OA',
        lexicon=_OA_ANCHOR,
        locator=r'\bmedial\b|\bmedially\b|\btricompartmental\b',
    ),
    TargetSpec(
        name='Lateral OA',
        lexicon=_OA_ANCHOR,
        locator=r'\blateral\b|\blaterally\b|\btricompartmental\b',
    ),
    TargetSpec(
        name='PF OA',
        lexicon=_OA_ANCHOR,
        locator=r'\bpatellofemoral\b|\bpf\s+(?:compartment|joint|space)\b|\btricompartmental\b',
    ),
    TargetSpec(
        name='Effusion',
        lexicon=r'\beffusions?\b|\bjoint\s+distension\b|\bhydrarthros\w*\b',
    ),
    TargetSpec(
        name='Synovitis',
        lexicon=(
            r'\bsynovitis\b'
            r'|\bsynovial\s+(?:thickening|enhancement|inflammation|proliferation)\b'
        ),
    ),
    TargetSpec(
        name="Baker's",
        lexicon=r"\bbakers?\s+cyst\b|\bbaker'?s?\b|\bpopliteal\s+cyst\b",
    ),
    TargetSpec(
        name='Contusion',
        lexicon=r'\bcontusions?\b|\bbone\s+bruises?\b|\bmarrow\s+edema\b',
    ),
    TargetSpec(
        name='Fracture',
        lexicon=r'\bfractures?\b|\bcortical\s+break\b|\bfx\b',
    ),
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*")
_SENTENCE_RE = re.compile(r'[^.;]+')


@dataclass
class _TokenizedReport:
    """Tokenization preserving character offsets and sentence membership.

    Attributes:
        text: Lowercased original report.
        tokens: Word tokens in order.
        spans: Character ``(start, end)`` span per token.
        sentence_ids: Sentence index per token; triggers never cross
            sentence boundaries when classifying mentions.
    """

    text: str
    tokens: list[str]
    spans: list[tuple[int, int]]
    sentence_ids: list[int]


class RuleBasedLabeler:
    """Negation-aware keyword labeler emitting {0, 1, -1} per target."""

    def __init__(
        self,
        specs: tuple[TargetSpec, ...] = TARGET_SPECS,
        negation_triggers: tuple[str, ...] = DEFAULT_NEGATION_TRIGGERS,
        uncertain_triggers: tuple[str, ...] = DEFAULT_UNCERTAIN_TRIGGERS,
        negation_window: int = 6,
    ) -> None:
        """Configure the labeler.

        Args:
            specs: Target specifications (defaults cover all 12 targets).
            negation_triggers: Tokens flipping a nearby mention to negative.
            uncertain_triggers: Tokens flipping a nearby mention to unknown.
            negation_window: Number of preceding tokens scanned for triggers.
        """
        self.specs = specs
        self.negation_triggers = frozenset(negation_triggers)
        self.uncertain_triggers = frozenset(uncertain_triggers)
        self.negation_window = negation_window

    @staticmethod
    def _tokenize(text: str) -> _TokenizedReport:
        """Lowercase and tokenize a report keeping offsets and sentences.

        Args:
            text: Raw report string.

        Returns:
            Tokenization container with tokens, character spans, and the
            sentence index assigned to every token.
        """
        lowered = str(text).lower()
        sentence_ranges = [m.span() for m in _SENTENCE_RE.finditer(lowered)]
        tokens: list[str] = []
        spans: list[tuple[int, int]] = []
        sentence_ids: list[int] = []
        cursor = 0
        for match in _TOKEN_RE.finditer(lowered):
            start, end = match.start(), match.end()
            while cursor < len(sentence_ranges) and start >= sentence_ranges[cursor][1]:
                cursor += 1
            tokens.append(match.group(0))
            spans.append((start, end))
            sentence_ids.append(cursor)
        return _TokenizedReport(
            text=lowered,
            tokens=tokens,
            spans=spans,
            sentence_ids=sentence_ids,
        )

    def _span_token_range(
        self,
        report: _TokenizedReport,
        start_char: int,
        end_char: int,
    ) -> tuple[int, int]:
        """Map a character span onto inclusive token indices.

        Args:
            report: Tokenized report.
            start_char: Span start offset in report text.
            end_char: Span end offset in report text.

        Returns:
            ``(first_token, last_token)`` indices clamped to the vocabulary.
        """
        first = next(
            (i for i, (s, _) in enumerate(report.spans) if s >= start_char),
            max(len(report.spans) - 1, 0),
        )
        last = next(
            (i for i, (_, e) in enumerate(report.spans) if e >= end_char),
            max(len(report.spans) - 1, 0),
        )
        return first, max(last, first)

    def _classify_span(
        self,
        report: _TokenizedReport,
        start_char: int,
        end_char: int,
    ) -> int:
        """Classify one mention scanning both sides for the nearest trigger.

        Radiology places negators both before structures ("no ACL tear") and
        as trailing predicates ("the ACL is intact"), so the window extends
        symmetrically; the closest trigger decides, and negation outranks
        uncertainty at equal distance.

        Args:
            report: Tokenized report containing the mention.
            start_char: Mention start offset.
            end_char: Mention end offset.

        Returns:
            POSITIVE / NEGATED / UNKNOWN verdict for the mention.
        """
        first, last = self._span_token_range(report, start_char, end_char)
        sentence = report.sentence_ids[first] if first < len(report.sentence_ids) else -1
        candidates: list[tuple[int, int]] = []  # (distance, kind)
        for distance in range(1, self.negation_window + 1):
            left = first - distance
            if left >= 0 and report.sentence_ids[left] == sentence:
                token = report.tokens[left]
                if token in self.negation_triggers:
                    candidates.append((distance, NEGATED))
                elif token in self.uncertain_triggers:
                    candidates.append((distance, UNKNOWN))
            right = last + distance
            if right < len(report.tokens) and report.sentence_ids[right] == sentence:
                token = report.tokens[right]
                if token in self.negation_triggers:
                    candidates.append((distance, NEGATED))
                elif token in self.uncertain_triggers:
                    candidates.append((distance, UNKNOWN))
            if candidates and all(c[0] <= distance for c in candidates):
                break
        if not candidates:
            return POSITIVE
        candidates.sort(key=lambda item: (item[0], 0 if item[1] == NEGATED else 1))
        return candidates[0][1]

    def _verdicts_for_pattern(self, report: _TokenizedReport, pattern: str) -> list[int]:
        """Classify every lexical match of a simple-target pattern.

        Args:
            report: Tokenized report.
            pattern: Affirmative regex string.

        Returns:
            Verdict per matched mention.
        """
        return [
            self._classify_span(report, match.start(), match.end())
            for match in re.finditer(pattern, report.text)
        ]

    def _verdicts_cooccurrence(
        self,
        report: _TokenizedReport,
        anchor: str,
        locator: str,
        radius: int,
    ) -> list[int]:
        """Classify anchor x locator proximity pairs for compartment OA.

        Matching runs over raw-text character spans (anchors such as
        ``cartilage loss`` span multiple tokens), pairing matches whose gap
        does not exceed ``radius`` average-token-lengths. A pair inherits
        the worse of its two mentions' classifications.

        Args:
            report: Tokenized report.
            anchor: Anchor regex (OA vocabulary).
            locator: Compartment regex (medial / lateral / patellofemoral).
            radius: Maximum inter-match distance in average token lengths.

        Returns:
            Verdict per qualifying pair.
        """
        mean_token_len = (
            sum(e - s for s, e in report.spans) / max(len(report.spans), 1)
        )
        max_gap = max(radius * mean_token_len, 3.0 * radius)
        anchors = [(m.start(), m.end()) for m in re.finditer(anchor, report.text)]
        locators = [(m.start(), m.end()) for m in re.finditer(locator, report.text)]
        verdicts: list[int] = []
        for a_start, a_end in anchors:
            for l_start, l_end in locators:
                gap = max(l_start - a_end, a_start - l_end)
                if gap > max_gap:
                    continue
                verdict_a = self._classify_span(report, a_start, a_end)
                verdict_l = self._classify_span(report, l_start, l_end)
                if NEGATED in (verdict_a, verdict_l):
                    verdicts.append(NEGATED)
                elif UNKNOWN in (verdict_a, verdict_l):
                    verdicts.append(UNKNOWN)
                else:
                    verdicts.append(POSITIVE)
        return verdicts

    @staticmethod
    def _aggregate(verdicts: list[int]) -> int:
        """Collapse per-mention verdicts into one target label.

        Args:
            verdicts: Collected verdicts for one target in one report.

        Returns:
            1 when any affirmation exists, else -1 when uncertainty exists,
            else 0 when everything is negated, else -1 (never mentioned).
        """
        if not verdicts:
            return UNKNOWN
        if POSITIVE in verdicts:
            return POSITIVE
        if UNKNOWN in verdicts:
            return UNKNOWN
        return NEGATED

    def label_report(self, text: str) -> dict[str, int]:
        """Label one report across every configured target.

        Args:
            text: Raw report text in any language (English rules apply).

        Returns:
            Mapping target -> {0, 1, -1}; -1 marks unknown/unmentioned.
        """
        report = self._tokenize(text)
        labels: dict[str, int] = {}
        for spec in self.specs:
            if spec.locator is None:
                verdicts = self._verdicts_for_pattern(report, spec.lexicon or '')
            else:
                verdicts = self._verdicts_cooccurrence(
                    report,
                    spec.lexicon or '',
                    spec.locator,
                    spec.cooccurrence_window,
                )
            labels[spec.name] = self._aggregate(verdicts)
        return labels


def build_pseudo_labels(
    train_df: pd.DataFrame,
    study_column: str,
    target_columns: list[str],
    labeler: RuleBasedLabeler | None = None,
) -> pd.DataFrame:
    """Derive rule-based labels for every study, preferring gold where present.

    Args:
        train_df: Frame holding StudyInstanceUID, Report, and target columns
            whose NaN cells indicate missing supervision.
        study_column: Name of the study identifier column.
        target_columns: The twelve canonical target names.
        labeler: Optional pre-configured labeler instance.

    Returns:
        DataFrame with ``StudyInstanceUID``, twelve label columns (gold value
        when available, else rule output), ``source`` ('gold' | 'rules'),
        and ``<target>_rule`` QC columns carrying raw rule outputs.
    """
    labeler = labeler or RuleBasedLabeler()
    rows: list[dict] = []
    rule_targets = [spec.name for spec in labeler.specs]
    for record in train_df.itertuples(index=False):
        row = record._asdict()
        rule_labels = labeler.label_report(row.get('Report') or '')
        entry: dict = {study_column: row.get(study_column)}
        sources = []
        for target in target_columns:
            gold_value = row.get(target, np.nan)
            rule_value = rule_labels.get(target, UNKNOWN)
            entry[f'{target}_rule'] = rule_value
            if pd.notna(gold_value):
                entry[target] = int(gold_value)
                sources.append('gold')
            else:
                entry[target] = rule_value
                sources.append('rules')
        entry['source'] = 'gold' if 'gold' in sources else 'rules'
        rows.append(entry)
    return pd.DataFrame(rows, columns=[study_column] + target_columns + ['source']
                        + [f'{target}_rule' for target in rule_targets])
