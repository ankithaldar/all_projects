#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the logging facility.

Four properties are load-bearing here and each has a positive control
beside it, because a facility that cannot be trusted to stay quiet is
worse than no facility:

  * **Idempotence.** Two calls to ``configure`` must emit one line. The
    failure mode this guards is doubled output, which is easy to miss by
    eye and makes every other assertion in a log-reading test ambiguous.
  * **Silence costs nothing.** A call below the threshold must not copy
    a field, render a value or read a clock. The tests use objects that
    *count* the work asked of them, so "no line" cannot pass by accident.
  * **Redaction.** A credential must not appear in captured output and
    a non-sensitive value must, checked in the same test class so a rule
    that started redacting everything would fail rather than pass.
  * **No raising, no mutation.** A ``__str__`` that raises, a
    self-referential dict and a handler that raises are all non-events
    from the caller's point of view, and a logged dict must still be the
    caller's dict afterwards.
'''

# pytest injects a fixture by parameter name, so every test method that
# asks for ``sink`` rebinds a module-level name. That is pytest's calling
# convention rather than a shadowing mistake, and the alternative --
# naming the fixture and the parameter differently -- makes these tests
# harder to read than the warning is worth.
# pylint: disable=redefined-outer-name

from __future__ import annotations

import ast
import io
import logging
import shlex
import sys
from collections import defaultdict

import pytest

from stock_rl import logging_support
from stock_rl.logging_support import (DEFAULT_FORMAT, LEVEL_ENV_VAR,
                                      MAX_FIELDS, REDACTED,
                                      ROOT_LOGGER_NAME, UNRENDERABLE,
                                      configure, debug_step, get_logger,
                                      is_sensitive, log_event, redact,
                                      redact_fields)

#: Keys this project logs that must survive redaction untouched. Every
#: entry here has been checked against :data:`SENSITIVE_SUBSTRINGS`,
#: which is why 'mapping' and 'bypass' are in it: they are the two words
#: that make a naive substring rule unusable.
NON_SENSITIVE_KEYS = (
  'symbol',
  'quantity',
  'price',
  'weight',
  'elapsed_s',
  'ok',
  'mapping',
  'bypass',
  'shipping',
  'client_order_ref',
)


@pytest.fixture(autouse=True)
def _pristine_logging():
  '''Restore the logging tree after every test in this file.

  Yields:
    Nothing; the fixture is for its teardown.
  '''
  root = logging.getLogger()
  before = list(root.handlers)
  project = logging.getLogger(ROOT_LOGGER_NAME)
  level = project.level
  yield
  for handler in list(root.handlers):
    if handler not in before:
      root.removeHandler(handler)
      handler.close()
  project.setLevel(level)


@pytest.fixture
def sink():
  '''Return a buffer that is this project's only log destination.

  The format is the bare message so that a captured line can be parsed
  as fields without stripping a timestamp first.

  Yields:
    An :class:`io.StringIO` receiving every record.
  '''
  target = io.StringIO()
  configure(logging.DEBUG, stream=target, fmt='%(message)s')
  yield target


def lines(target: io.StringIO) -> list[str]:
  '''Return the non-empty lines written to a buffer.

  Args:
    target: The buffer.

  Returns:
    One string per record.
  '''
  return [line for line in target.getvalue().splitlines() if line]


def parse(line: str) -> dict[str, str]:
  '''Parse one emitted line back into its fields.

  Args:
    line: A line produced by :func:`log_event`.

  Returns:
    The ``key=value`` pairs as a mapping. This is the round trip that
    makes the output greppable rather than merely printed.
  '''
  parsed: dict[str, str] = {}
  for token in shlex.split(line):
    key, _, value = token.partition('=')
    parsed[key] = value
  return parsed


class _Counter:
  '''Counts the rendering work a log call asked of it.'''

  def __init__(self):
    self.strs = 0
    self.reprs = 0

  def __str__(self):
    self.strs += 1
    return 'counter'

  def __repr__(self):
    self.reprs += 1
    return 'counter'


class _NoStr:
  '''An object whose ``str`` raises but whose ``repr`` does not.'''

  def __str__(self):
    raise RuntimeError('this object refuses str')

  def __repr__(self):
    return '_NoStr()'


class _Hostile:
  '''An object that refuses both text conversions.'''

  def __str__(self):
    raise RuntimeError('this object refuses str')

  def __repr__(self):
    raise RuntimeError('this object also refuses repr')


class _ExplodingHandler(logging.Handler):
  '''A sink that raises, standing in for a broken stream or a full disk.'''

  def emit(self, record):
    del record
    raise RuntimeError('the sink is broken')


class _UnreadableKey:
  '''A field name that cannot be turned into text.'''

  def __str__(self):
    raise RuntimeError('this key refuses to be named')

  def __repr__(self):
    return '_UnreadableKey()'


class _UnreadableDict(dict):
  '''A mapping that cannot be walked, though it still is one.'''

  def items(self):
    raise ValueError('this mapping refuses to be read')


class _BrokenLogger:
  '''A logger whose level check raises, as a misconfigured one might.'''

  # The camelCase name is the stdlib's, and log_event is duck-typed on
  # it, so a snake_case spelling here would not be testing the path.
  def isEnabledFor(self, level):  # pylint: disable=invalid-name
    del level
    raise RuntimeError('this logger is broken')

  def log(self, level, message, *args):
    del level, message, args
    raise AssertionError('log must not be reached')


class TestConfigureIsIdempotent:
  '''Repeated configuration must not double every line.'''

  def test_two_calls_emit_one_line(self, sink):
    configure(logging.DEBUG, stream=sink, fmt='%(message)s')
    configure(logging.DEBUG, stream=sink, fmt='%(message)s')
    log_event(get_logger('weights'), 'rebalance', symbols=12)
    assert len(lines(sink)) == 1

  def test_positive_control_one_call_also_emits_one_line(self, sink):
    # The neighbour of the test above. If a single configure emitted
    # nothing, the idempotence test would pass for the wrong reason.
    log_event(get_logger('weights'), 'rebalance', symbols=12)
    assert len(lines(sink)) == 1

  def test_second_call_replaces_the_first_destination(self):
    first, second = io.StringIO(), io.StringIO()
    configure(logging.DEBUG, stream=first, fmt='%(message)s')
    configure(logging.DEBUG, stream=second, fmt='%(message)s')
    log_event(get_logger('weights'), 'rebalance')
    assert not first.getvalue()
    assert 'event=rebalance' in second.getvalue()

  def test_a_handler_this_module_did_not_install_is_left_alone(self):
    foreign = logging.NullHandler()
    logging.getLogger().addHandler(foreign)
    configure(logging.DEBUG, stream=io.StringIO())
    assert foreign in logging.getLogger().handlers

  def test_configure_returns_the_project_logger(self):
    assert configure(logging.INFO, stream=io.StringIO()) is logging.getLogger(
      ROOT_LOGGER_NAME)

  def test_the_default_format_names_time_level_and_module(self):
    # A line that cannot say which layer produced it forces the reader to
    # guess, which is the reading this module exists to remove.
    for token in ('asctime', 'levelname', 'name', 'message'):
      assert token in DEFAULT_FORMAT

  def test_the_default_destination_is_stderr(self):
    configure(logging.INFO)
    installed = [
      handler for handler in logging.getLogger().handlers
      if getattr(handler, logging_support.HANDLER_TAG, None)
    ]
    assert [handler.stream for handler in installed] == [sys.stderr]

  def test_level_comes_from_the_environment(self, monkeypatch):
    monkeypatch.setenv(LEVEL_ENV_VAR, 'WARNING')
    logger = configure(stream=io.StringIO())
    assert logger.level == logging.WARNING

  def test_positive_control_the_environment_variable_is_optional(
      self, monkeypatch):
    # Without the variable the default applies, so the test above cannot
    # be passing because every level resolved to WARNING.
    monkeypatch.delenv(LEVEL_ENV_VAR, raising=False)
    assert configure(stream=io.StringIO()).level == logging.INFO

  def test_a_misspelled_level_is_refused_loudly(self):
    with pytest.raises(ValueError, match='unknown log level'):
      configure('VERBOSE', stream=io.StringIO())

  def test_positive_control_a_known_level_name_is_accepted(self):
    assert configure('debug', stream=io.StringIO()).level == logging.DEBUG

  def test_a_numeric_level_is_accepted(self):
    assert configure(logging.ERROR, stream=io.StringIO()).level == (
      logging.ERROR)

  def test_bool_is_refused_as_a_level(self):
    # bool is an int and True would configure the whole project at
    # CRITICAL, which looks like a run that logs nothing at all.
    with pytest.raises(ValueError, match='level must be'):
      configure(True, stream=io.StringIO())


class TestGetLogger:
  '''Every logger this project hands out shares one namespace.'''

  def test_a_short_name_is_namespaced(self):
    assert get_logger('weights').name == 'stock_rl.weights'

  def test_a_full_name_passes_through(self):
    # A caller holding __name__ will pass it; requiring it to be stripped
    # first would make every call site strip it first.
    assert get_logger('stock_rl.rl.train').name == 'stock_rl.rl.train'

  def test_no_name_returns_the_namespace_itself(self):
    assert get_logger().name == ROOT_LOGGER_NAME

  def test_positive_control_it_is_the_standard_library_logger(self):
    assert get_logger('weights') is logging.getLogger('stock_rl.weights')

  def test_the_project_level_gates_a_child_logger(self):
    # No level set on the child: this is the mechanism that makes
    # configure() a single switch for the whole project.
    logger = get_logger('portfolio')
    assert not logger.level
    configure(logging.WARNING, stream=io.StringIO())
    assert not logger.isEnabledFor(logging.INFO)


class TestSilenceCostsNothing:
  '''A suppressed call must not do the work of an emitted one.'''

  def test_a_suppressed_call_renders_nothing(self):
    target = io.StringIO()
    configure(logging.WARNING, stream=target, fmt='%(message)s')
    counter = _Counter()
    emitted = log_event(get_logger('weights'), 'rebalance',
                        level=logging.DEBUG, weight=counter)
    assert emitted is False
    assert counter.strs == 0
    assert counter.reprs == 0
    assert not lines(target)

  def test_positive_control_an_enabled_call_renders(self):
    target = io.StringIO()
    configure(logging.DEBUG, stream=target, fmt='%(message)s')
    counter = _Counter()
    emitted = log_event(get_logger('weights'), 'rebalance',
                        weight=counter)
    assert emitted is True
    assert counter.strs + counter.reprs > 0
    assert 'weight=counter' in lines(target)[0]

  def test_a_suppressed_call_does_not_walk_a_mapping(self):
    class _Exploding:
      '''A mapping that cannot be walked.'''

      def items(self):
        raise AssertionError('the field was walked while suppressed')

    configure(logging.WARNING, stream=io.StringIO(), fmt='%(message)s')
    assert log_event(get_logger('weights'), 'rebalance',
                     positions=_Exploding()) is False

  def test_positive_control_an_enabled_call_walks_the_mapping(self):
    # The neighbour of the test above: an enabled call must reach the
    # mapping, or the suppression above would prove nothing.
    target = io.StringIO()
    configure(logging.DEBUG, stream=target, fmt='%(message)s')
    positions = {'RELIANCE': 10, 'TCS': 5}
    assert log_event(get_logger('weights'), 'rebalance',
                     positions=positions) is True
    assert 'RELIANCE' in target.getvalue()


class TestRedaction:
  '''A credential must never reach a log; a metric always may.'''

  def test_a_credential_never_appears_in_output(self, sink):
    secret = 'X7QK2LMNP4WZ6RTE8YGH'
    log_event(get_logger('execution'), 'broker_auth',
              api_key=secret, symbol='RELIANCE')
    output = sink.getvalue()
    assert secret not in output
    assert REDACTED in output

  def test_a_non_sensitive_value_does_appear(self, sink):
    # The positive control for the test above. A rule that redacted
    # everything would pass it, and a log with no numbers in it is the
    # other way this facility becomes useless.
    log_event(get_logger('execution'), 'broker_auth',
              symbol='RELIANCE', quantity=100, average_price=2543.75)
    output = sink.getvalue()
    assert 'symbol=RELIANCE' in output
    assert 'quantity=100' in output
    assert 'average_price=2543.75' in output

  @pytest.mark.parametrize('key', sorted(logging_support.SENSITIVE_KEYS))
  def test_every_declared_sensitive_key_is_redacted(self, key, sink):
    log_event(get_logger('execution'), 'field_probe', **{key: 'PROBEVALUE'})
    assert 'PROBEVALUE' not in sink.getvalue()

  @pytest.mark.parametrize('key', NON_SENSITIVE_KEYS)
  def test_keys_this_project_actually_logs_are_not_redacted(self, key,
                                                            sink):
    log_event(get_logger('portfolio'), 'field_probe', **{key: 'PROBEVALUE'})
    assert 'PROBEVALUE' in sink.getvalue()

  def test_an_nnf_id_is_caught_under_an_innocuous_key(self, sink):
    # The case a key-name rule alone cannot see coming: 15 digits of
    # NNF ID arriving as 'reference'.
    nnf = '4444444444442' + '00'
    log_event(get_logger('execution'), 'order_sent', reference=nnf)
    assert nnf not in sink.getvalue()
    assert REDACTED in sink.getvalue()

  def test_a_client_direct_platform_prefix_is_caught(self, sink):
    # From stock_rl.compliance.algo_tag: a CTCL prefix's first six digits
    # are the client PIN, so a 12-digit platform prefix is an identifier
    # whatever it is named.
    prefix = '123456123456'
    log_event(get_logger('execution'), 'platform_probe', venue=prefix)
    assert prefix not in sink.getvalue()

  def test_a_short_digit_run_survives(self, sink):
    # The positive control for the value-shape rule: a date is eight
    # digits and must not be redacted, or the rule is useless.
    log_event(get_logger('bars'), 'bar_loaded', session='20260101')
    assert 'session=20260101' in sink.getvalue()

  def test_a_redacted_value_is_still_logged_under_its_key(self, sink):
    # The key is the diagnosis. A dropped field leaves no evidence that
    # it was ever present, and an operator debugging an attribution
    # question needs to know the field arrived at all.
    log_event(get_logger('execution'), 'order_sent', algo_id='ALGO-7')
    assert parse(lines(sink)[0])['algo_id'] == REDACTED

  def test_nested_mapping_keys_are_redacted(self, sink):
    log_event(get_logger('execution'), 'batch',
              orders=[{'symbol': 'TCS', 'order_id': 'BROKER-99'}])
    output = sink.getvalue()
    assert 'BROKER-99' not in output
    assert 'TCS' in output

  def test_extra_keys_extend_the_rules_for_one_call(self, sink):
    log_event(get_logger('execution'), 'seat_probe',
              seat_token='SEAT-77', extra_keys=('seat_token',))
    assert 'SEAT-77' not in sink.getvalue()

  def test_positive_control_extra_keys_do_not_leak_into_the_next_call(
      self, sink):
    # The extension is a per-call argument, not a global registry. If it
    # were global, this second call would still redact the value.
    log_event(get_logger('execution'), 'seat_probe', seat_token='SEAT-77')
    assert 'SEAT-77' in sink.getvalue()

  def test_a_key_that_cannot_be_named_is_redacted(self):
    # Fails closed: a name this module cannot read is a name it cannot
    # clear. Note this cannot arrive through log_event -- **kwargs keys
    # must be strings -- so it is reached through the redaction helpers,
    # which is exactly why they are public.
    assert redact(_UnreadableKey(), 'PROBEVALUE') == REDACTED

  def test_positive_control_a_named_key_is_not_redacted(self, sink):
    # The neighbour of the test above. Without it, "every key that is not
    # a plain string is redacted" would satisfy both and the fail-closed
    # rule would be untested.
    log_event(get_logger('execution'), 'key_probe', seat='PROBE')
    assert 'PROBE' in sink.getvalue()

  def test_is_sensitive_normalises_case_and_dashes(self):
    assert is_sensitive('API-KEY')
    assert is_sensitive('  Algo_Id  ')

  def test_positive_control_is_sensitive_rejects_the_obvious_decoys(self):
    # 'pin' and 'pass' are exact keys only, because as substrings they
    # match 'mapping' and 'bypass' and would redact half this project.
    for key in NON_SENSITIVE_KEYS:
      assert not is_sensitive(key), key

  def test_the_redaction_helpers_are_usable_on_their_own(self):
    assert redact('order_id', 'BROKER-1') == REDACTED
    assert redact('symbol', 'TCS') == 'TCS'
    assert redact_fields({'api_key': 'x', 'symbol': 'TCS'}) == {
      'api_key': REDACTED, 'symbol': 'TCS'}


class TestNeverRaises:
  '''No rendering path may raise into a caller.'''

  def test_an_object_whose_str_raises_does_not_propagate(self, sink):
    assert log_event(get_logger('weights'), 'probe',
                     weight=_NoStr()) is True
    assert '_NoStr()' in sink.getvalue()

  def test_an_object_whose_repr_also_raises_does_not_propagate(
      self, sink):
    assert log_event(get_logger('weights'), 'probe',
                     weight=_Hostile()) is True
    assert UNRENDERABLE in sink.getvalue()

  def test_positive_control_an_ordinary_object_is_rendered_in_full(
      self, sink):
    assert log_event(get_logger('weights'), 'probe',
                     weight=3.5) is True
    assert 'weight=3.5' in sink.getvalue()

  def test_a_self_referential_dict_terminates(self, sink):
    # A dict that contains itself is constructible in one line and
    # unbounded to render without a depth limit. The cycle shows up as
    # the truncated marker inside the rendered copy.
    loop: dict[str, object] = {}
    loop['self'] = loop
    assert log_event(get_logger('weights'), 'probe', loop=loop) is True
    assert loop['self'] is loop
    rendered = ast.literal_eval(parse(lines(sink)[0])['loop'])
    assert rendered == {'self': '<cycle>'}

  def test_a_positive_control_nested_dict_renders_in_full(self, sink):
    # The neighbour of the test above. Without it, "every nested
    # structure is replaced by a marker" would satisfy both.
    log_event(get_logger('weights'), 'probe', book={'a': {'b': 1}})
    rendered = ast.literal_eval(parse(lines(sink)[0])['book'])
    assert rendered == {'a': {'b': 1}}

  def test_a_container_deeper_than_the_limit_is_truncated(self, sink):
    deep = {'l1': {'l2': {'l3': {'l4': {'l5': 'bottom'}}}}}
    log_event(get_logger('weights'), 'probe', deep=deep)
    rendered = ast.literal_eval(parse(lines(sink)[0])['deep'])
    assert '...' in repr(rendered)
    assert 'bottom' not in repr(rendered)

  def test_a_handler_that_raises_is_swallowed(self):
    target = io.StringIO()
    configure(logging.DEBUG, stream=target, fmt='%(message)s')
    logging.getLogger().addHandler(_ExplodingHandler())
    assert log_event(get_logger('weights'), 'probe') is False

  def test_a_logger_that_raises_does_not_take_down_the_caller(self):
    # The barrier is in the public entry point, not only in the emitter,
    # so a misconfigured logger is survivable too.
    assert log_event(_BrokenLogger(), 'probe', weight=1.0) is False

  def test_a_mapping_that_cannot_be_walked_becomes_one_marker(self, sink):
    # One unreadable value must not cost the caller the rest of the line.
    assert log_event(get_logger('weights'), 'probe',
                     before=1, payload=_UnreadableDict(a=1),
                     after=2) is True
    record = parse(lines(sink)[0])
    assert record['payload'] == UNRENDERABLE
    assert record['before'] == '1'
    assert record['after'] == '2'

  def test_a_set_renders_sorted(self, sink):
    # Sorted, so two runs of the same code produce the same line; a set's
    # own iteration order depends on hash randomisation and would make
    # every run look like a change.
    log_event(get_logger('portfolio'), 'probe', symbols={'TCS', 'INFY'})
    assert parse(lines(sink)[0])['symbols'] == "['INFY', 'TCS']"

  def test_a_broken_handler_does_not_break_a_step(self):
    target = io.StringIO()
    configure(logging.DEBUG, stream=target, fmt='%(message)s')
    logging.getLogger().addHandler(_ExplodingHandler())
    with debug_step('run_portfolio', get_logger('portfolio')) as step:
      step.set(turnover=0.5)
    assert step.elapsed_s >= 0.0

  def test_an_unserialisable_value_is_truncated_rather_than_lost(self,
                                                                 sink):
    log_event(get_logger('weights'), 'probe', blob='x' * 5000)
    body = lines(sink)[0]
    assert '...' in body
    assert len(body) < 400


class TestNoMutation:
  '''A log call must leave the caller's data exactly as it found it.'''

  def test_a_dict_field_is_unchanged_afterwards(self, sink):
    weights = {'RELIANCE': 0.4, 'TCS': 0.3}
    snapshot = dict(weights)
    log_event(get_logger('weights'), 'rebalance', weights=weights)
    assert 'RELIANCE' in sink.getvalue()
    assert weights == snapshot

  def test_a_nested_dict_is_replaced_by_a_copy_not_edited(self, sink):
    positions = {'RELIANCE': 10}
    log_event(get_logger('portfolio'), 'snapshot', book=positions)
    logged = ast.literal_eval(parse(lines(sink)[0])['book'])
    assert logged is not positions
    assert logged == positions

  def test_a_list_field_is_unchanged_afterwards(self, sink):
    reasons = ['filled', 'partially filled']
    snapshot = list(reasons)
    log_event(get_logger('execution'), 'acks', reasons=reasons)
    assert 'partially filled' in sink.getvalue()
    assert reasons == snapshot

  def test_a_nested_defaultdict_stays_a_defaultdict_for_the_caller(
      self, sink):
    # The copy this module renders is a plain dict. If it were rendered
    # by mutating the original, a defaultdict would lose its factory the
    # first time it was logged and the caller would get a KeyError in a
    # place no test was looking.
    grouped = defaultdict(list)
    grouped['RELIANCE'].append(10)
    log_event(get_logger('portfolio'), 'snapshot', grouped=grouped)
    assert 'RELIANCE' in sink.getvalue()
    assert grouped.default_factory is list
    assert grouped == {'RELIANCE': [10]}

  def test_logging_twice_produces_the_same_line(self, sink):
    # The strongest available control on mutation: if the first call had
    # scrubbed the caller's dict in place, the second line would differ.
    weights = {'RELIANCE': 0.4, 'TCS': 0.3}
    log_event(get_logger('weights'), 'rebalance', weights=weights)
    first = lines(sink)[-1]
    log_event(get_logger('weights'), 'rebalance', weights=weights)
    assert lines(sink)[-1] == first

  def test_a_step_does_not_edit_the_fields_it_was_given(self, sink):
    fields = {'symbols': 3, 'order_id': 'BROKER-1'}
    snapshot = dict(fields)
    with debug_step('run_portfolio', get_logger('portfolio'), **fields):
      pass
    assert fields == snapshot
    assert 'symbols=3' in sink.getvalue()
    assert 'BROKER-1' not in sink.getvalue()

  def test_positive_control_the_log_actually_contains_the_data(self,
                                                                sink):
    # Otherwise every assertion above would pass on an empty log.
    log_event(get_logger('weights'), 'rebalance', weights={'TCS': 0.3})
    assert 'TCS' in sink.getvalue()


class TestStep:
  '''The context manager must bracket an operation on both sides.'''

  def test_both_boundaries_are_logged(self, sink):
    with debug_step('run_portfolio', get_logger('portfolio'), symbols=3):
      pass
    assert [parse(line)['event'] for line in lines(sink)] == [
      'step.begin', 'step.end']

  def test_the_elapsed_time_is_non_negative(self, sink):
    with debug_step('run_portfolio', get_logger('portfolio')) as step:
      pass
    assert step.elapsed_s >= 0.0
    assert float(parse(lines(sink)[-1])['elapsed_s']) >= 0.0

  def test_the_opening_line_carries_the_inputs(self, sink):
    with debug_step('run_portfolio', get_logger('portfolio'), symbols=3):
      pass
    opening = parse(lines(sink)[0])
    assert opening['step'] == 'run_portfolio'
    assert opening['symbols'] == '3'

  def test_results_land_on_the_closing_line_only(self, sink):
    with debug_step('run_portfolio', get_logger('portfolio')) as step:
      step.set(turnover=0.31)
    assert 'turnover' not in lines(sink)[0]
    assert parse(lines(sink)[1])['turnover'] == '0.31'

  def test_a_reserved_field_name_cannot_be_clobbered(self, sink):
    # A caller field named 'step' must not erase the operation name, and
    # must not raise a duplicate-keyword TypeError inside __exit__ where
    # it would replace whatever exception the body raised.
    with debug_step('run_portfolio', get_logger('portfolio'),
                    step='not-the-name', elapsed_s=-1.0):
      pass
    closing = parse(lines(sink)[1])
    assert closing['step'] == 'run_portfolio'
    assert float(closing['elapsed_s']) >= 0.0

  def test_a_failing_body_records_the_type_and_still_raises(self, sink):
    with pytest.raises(ValueError, match='body exploded'):
      with debug_step('size_order', get_logger('execution')):
        raise ValueError('body exploded')
    closing = parse(lines(sink)[-1])
    assert closing['ok'] == 'False'
    assert closing['error'] == 'ValueError'

  def test_a_failing_step_logs_no_exception_message(self, sink):
    # The message is free text this module cannot redact, so it is not
    # logged. This is the deliberate limit of the redaction, and a test is
    # the only place it can be stated honestly.
    with pytest.raises(RuntimeError):
      with debug_step('connect', get_logger('execution')):
        raise RuntimeError('api_key=SECRETVALUE rejected')
    output = sink.getvalue()
    assert 'SECRETVALUE' not in output
    assert 'RuntimeError' in output

  def test_a_disabled_step_writes_nothing(self):
    target = io.StringIO()
    configure(logging.INFO, stream=target, fmt='%(message)s')
    with debug_step('run_portfolio', get_logger('portfolio'),
                    symbols=3) as step:
      step.set(turnover=0.31)
    assert not lines(target)
    assert step.elapsed_s >= 0.0

  def test_positive_control_the_same_step_is_logged_when_enabled(self,
                                                                 sink):
    with debug_step('run_portfolio', get_logger('portfolio'),
                    symbols=3):
      pass
    assert len(lines(sink)) == 2

  def test_the_enabled_property_reflects_the_threshold(self):
    logger = get_logger('portfolio')
    configure(logging.INFO, stream=io.StringIO())
    assert debug_step('run_portfolio', logger).enabled is False
    configure(logging.DEBUG, stream=io.StringIO())
    assert debug_step('run_portfolio', logger).enabled is True

  def test_a_step_defaults_to_this_modules_logger(self):
    # A caller that names no logger still gets output, attributed to the
    # facility rather than to whatever last configured the root.
    assert debug_step('run_portfolio').logger.name.endswith(
      'logging_support')


class TestOutputIsGreppable:
  '''A line must parse back into the fields that produced it.'''

  def test_fields_round_trip_through_shlex(self, sink):
    log_event(get_logger('portfolio'), 'rebalance',
              symbol='RELIANCE', quantity=100, average_price=2543.75)
    assert parse(lines(sink)[0]) == {
      'event': 'rebalance',
      'symbol': 'RELIANCE',
      'quantity': '100',
      'average_price': '2543.75',
    }

  def test_a_value_containing_spaces_survives_the_round_trip(self, sink):
    log_event(get_logger('execution'), 'rejected',
              reason='no live order with id X')
    assert parse(lines(sink)[0])['reason'] == 'no live order with id X'

  def test_a_value_containing_a_quote_survives_the_round_trip(self,
                                                               sink):
    log_event(get_logger('execution'), 'rejected', reason="broker's view")
    assert parse(lines(sink)[0])['reason'] == "broker's view"

  def test_a_value_containing_an_equals_sign_survives(self, sink):
    # shlex treats '=' as safe, so this is the one separator the parser
    # must not split on a second time.
    log_event(get_logger('execution'), 'probe', query='a=b')
    assert parse(lines(sink)[0])['query'] == 'a=b'

  def test_fields_are_sorted_for_a_diffable_line(self, sink):
    log_event(get_logger('portfolio'), 'rebalance',
              zeta=1, alpha=2, mid=3)
    body = lines(sink)[0].split(' ', 1)[1]
    assert body == 'alpha=2 mid=3 zeta=1'

  def test_too_many_fields_are_counted_not_hidden(self, sink):
    log_event(get_logger('portfolio'), 'flood',
              **{f'f{index:03d}': index for index in range(MAX_FIELDS + 5)})
    record = parse(lines(sink)[0])
    assert record['dropped'] == '5'
    assert len(record) == MAX_FIELDS + 2

  def test_positive_control_a_bare_event_still_parses(self, sink):
    log_event(get_logger('portfolio'), 'flush')
    assert parse(lines(sink)[0]) == {'event': 'flush'}

  def test_the_level_is_chosen_per_call(self, sink):
    log_event(get_logger('portfolio'), 'quiet', level=logging.WARNING)
    assert 'event=quiet' in lines(sink)[0]

  def test_a_debug_event_is_suppressed_at_info(self):
    target = io.StringIO()
    configure(logging.INFO, stream=target, fmt='%(message)s')
    log_event(get_logger('portfolio'), 'chatter', level=logging.DEBUG)
    assert not lines(target)
    log_event(get_logger('portfolio'), 'decided')
    assert len(lines(target)) == 1

  def test_the_exported_surface_is_exactly_what_is_advertised(self):
    # __all__ is the contract; a name in it that does not exist is an
    # ImportError in somebody else's module.
    for name in logging_support.__all__:
      assert hasattr(logging_support, name), name
