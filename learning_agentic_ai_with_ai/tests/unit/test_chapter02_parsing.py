#!/usr/bin/env python
# -- coding: utf-8 --

'''Unit tests: defensive parsers for model output.'''


from __future__ import annotations

import pytest

from chapter02_planning.parsing import (
  ParseError,
  extract_json_block,
  extract_numbers,
  parse_cot_output,
  parse_critique_output,
  parse_json_into,
  parse_plan_output,
  parse_react_turn,
  parse_step_labeled,
)
from chapter02_planning.schemas import Critique, Plan


class TestExtractJsonBlock:
  '''JSON extraction across noisy real-world shapes.'''

  def test_raw_object(self) -> None:
    assert extract_json_block('{"a": 1}') == {'a': 1}

  def test_fenced_block_with_prose(self) -> None:
    text = 'Sure! Here it is:\n```json\n{"a": [1, 2]}\n```\nThanks.'
    assert extract_json_block(text) == {'a': [1, 2]}

  def test_untrusted_wrapper_span(self) -> None:
    text = '[BEGIN UNTRUSTED TOOL DATA]\n{"items": []}\n[END UNTRUSTED TOOL DATA]'
    assert extract_json_block(text) == {'items': []}

  def test_trailing_comma_repaired(self) -> None:
    assert extract_json_block('{"a": 1, "b": [2,],}') == {'a': 1, 'b': [2]}

  def test_smart_quotes_repaired(self) -> None:
    assert extract_json_block('{\u201ca\u201d: 1}') == {'a': 1}

  def test_python_literals_fallback(self) -> None:
    assert extract_json_block("{'a': True, 'b': None}") == {
      'a': True, 'b': None,
    }

  def test_unbalanced_raises(self) -> None:
    with pytest.raises(ParseError):
      extract_json_block('{"a": 1')

  def test_garbage_raises(self) -> None:
    with pytest.raises(ParseError):
      extract_json_block('no json here at all')

  def test_empty_raises(self) -> None:
    with pytest.raises(ParseError):
      extract_json_block('')

  def test_parse_into_model(self) -> None:
    critique = parse_json_into(
      '{"verdict": "pass", "issues": []}', Critique,
    )
    assert critique.verdict == 'pass'

  def test_parse_into_model_failure(self) -> None:
    with pytest.raises(ParseError):
      parse_json_into('{"verdict": "maybe"}', Critique)


class TestStepLabeled:
  '''Numbered and bulleted step parsing.'''

  def test_numbered_steps(self) -> None:
    text = '1. First thing\n2. Second thing\nAnswer: done'
    assert parse_step_labeled(text) == ['First thing', 'Second thing']

  def test_bulleted_steps(self) -> None:
    assert parse_step_labeled('- alpha\n* beta') == ['alpha', 'beta']

  def test_paragraph_fallback(self) -> None:
    assert parse_step_labeled('One block\n\nAnother block') == [
      'One block', 'Another block',
    ]


class TestCoTParser:
  '''Chain-of-thought output parsing.'''

  def test_answer_line(self) -> None:
    parsed = parse_cot_output(
      'Reasoning:\n1. Add 2 and 2\n2. Get 4\nAnswer: 4 units',
    )
    assert parsed.ok
    assert parsed.answer == '4 units'
    assert parsed.steps == ['Add 2 and 2', 'Get 4']

  def test_tagged_blocks(self) -> None:
    parsed = parse_cot_output(
      '<reasoning>1. do a thing</reasoning><answer>42</answer>',
    )
    assert parsed.ok
    assert parsed.answer == '42'

  def test_missing_answer_marker(self) -> None:
    parsed = parse_cot_output('I think the answer is probably 7')
    assert parsed.ok is False
    assert parsed.answer

  def test_empty_output(self) -> None:
    parsed = parse_cot_output('')
    assert parsed.ok is False


class TestReActParser:
  '''ReAct turn parsing.'''

  def test_action_with_json_input(self) -> None:
    parsed = parse_react_turn(
      'Thought: need stock\n'
      'Action: retail-ops.retail_low_stock_report\n'
      'Action Input: {"store_id": "S01"}',
    )
    assert parsed.ok
    assert parsed.action == 'retail-ops.retail_low_stock_report'
    assert parsed.action_input == {'store_id': 'S01'}
    assert parsed.is_final is False

  def test_final_answer(self) -> None:
    parsed = parse_react_turn(
      'Thought: done\nFinal Answer: order 110 units',
    )
    assert parsed.ok
    assert parsed.is_final
    assert parsed.final_answer == 'order 110 units'

  def test_missing_action(self) -> None:
    parsed = parse_react_turn('Thought: just thinking out loud')
    assert parsed.ok is False
    assert 'Action' in (parsed.error or '')

  def test_malformed_action_input(self) -> None:
    parsed = parse_react_turn(
      'Thought: x\nAction: tool\nAction Input: {bad json',
    )
    assert parsed.ok is False
    assert parsed.action == 'tool'

  def test_key_value_action_input(self) -> None:
    parsed = parse_react_turn(
      'Thought: x\nAction: tool\nAction Input: store_id=S01, days=7',
    )
    assert parsed.ok
    assert parsed.action_input == {'store_id': 'S01', 'days': 7}

  def test_empty_turn(self) -> None:
    parsed = parse_react_turn('   ')
    assert parsed.ok is False


class TestPlanParser:
  '''Plan JSON parsing and normalization.'''

  def test_valid_plan(self) -> None:
    plan = parse_plan_output(
      '```json\n'
      '{"goal": "g", "nodes": ['
      '{"id": "Step One", "objective": "do it", "depends_on": [],'
      ' "suggested_tools": ["t"], "success_criteria": "done"},'
      '{"id": "step_two", "objective": "more", "depends_on": ["step_one"]}'
      ']}\n```'
    )
    assert isinstance(plan, Plan)
    assert [node.id for node in plan.nodes] == ['step_one', 'step_two']
    assert plan.nodes[1].depends_on == ['step_one']

  def test_unknown_dependency_rejected(self) -> None:
    with pytest.raises(ParseError):
      parse_plan_output(
        '{"nodes": [{"id": "a", "objective": "x",'
        ' "depends_on": ["missing"]}]}'
      )

  def test_missing_nodes_rejected(self) -> None:
    with pytest.raises(ParseError):
      parse_plan_output('{"goal": "g", "nodes": []}')

  def test_plan_key_and_synonym_fields(self) -> None:
    '''Real models wrap nodes under "plan" and rename node fields.'''
    plan = parse_plan_output(
      '{"plan": ['
      '{"id": "1", "description": "fetch stock", "dependencies": []},'
      '{"id": "2", "description": "size the order",'
      ' "dependencies": ["1"]}'
      ']}'
    )
    assert [node.id for node in plan.nodes] == ['step_1', 'step_2']
    assert plan.nodes[0].objective == 'fetch stock'
    assert plan.nodes[1].depends_on == ['step_1']

  def test_top_level_list_payload(self) -> None:
    plan = parse_plan_output(
      '[{"id": "only", "task": "do it"}]'
    )
    assert plan.nodes[0].objective == 'do it'


class TestNumbers:
  '''Numeric token extraction.'''

  def test_numbers_with_commas(self) -> None:
    assert extract_numbers('we sold 1,200 units and 30.5% more') == {
      1200.0, 30.5,
    }

  def test_identifiers_not_numbers(self) -> None:
    assert 1.0 not in extract_numbers('store S01 sku R-101')

  def test_critique_parse(self) -> None:
    critique = parse_critique_output(
      '{"verdict": "fail", "issues": [{"check": "grounding",'
      ' "severity": "error", "detail": "bad"}], "revised_answer": "fixed"}'
    )
    assert critique.verdict == 'fail'
    assert critique.revised_answer == 'fixed'
