#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Locked bot session control flow (plan v11 section 6).

The implementation follows the approved control flow verbatim: a three-phase
gift protocol with both failure branches, a single general action per tick,
the full stop vocabulary, and deterministic cleanup that always closes the
database.  No device action occurs in dry-run mode.
"""

import hashlib
import json
import time
from dataclasses import asdict

from cat_game_bot.config import _json_default
from cat_game_bot.errors import AdbError, DbError, VisionError
from cat_game_bot.models import (
  Action, ActionKind, ActionResult, CraftStep, GameState, SessionStats,
)
from cat_game_bot.navigator import Memory
from cat_game_bot.targets import TARGET_IDS, goal_met


class Session:
  """Drives one bot run through the locked section-6 control flow."""

  _step: CraftStep | None
  _step_remaining: int

  def __init__(self, adb, detector, navigator, ocr, inventory, planner,
               gift, db, config, targets, now=time.time):
    """Inject all collaborators and initialise per-run state.

    Args:
        adb: ``AdbClient`` for screenshots and device actions.
        detector: ``ScreenDetector`` producing detections (contract-mandated
          injection; held for the graph even though detection is performed
          through ``settle``).
        navigator: ``Navigator`` selecting actions.
        ocr: ``OcrReader`` (contract-mandated injection; held for symmetry
          with the injected graph, not read directly by the session).
        inventory: ``InventoryReader`` reading Bag counts.
        planner: ``CraftPlanner`` selecting the next craft step.
        gift: ``GiftCollector`` handling the gift protocol.
        db: ``InteractionLog`` recording the run.
        config: Frozen ``Config``.
        targets: ``TargetSpec`` with the eleven target counts.
        now: Injectable clock returning epoch seconds.
    """
    self.adb = adb
    self.detector = detector
    self.navigator = navigator
    self.ocr = ocr
    self.inventory = inventory
    self.planner = planner
    self.gift = gift
    self.db = db
    self.config = config
    self.targets = targets
    self.now = now
    self.memory = Memory()
    self.counts = {}
    self.session_id = -1
    self._pending_gift_before = None
    self._gift_collect_tapped = False

  def settle(self):
    """Capture a settled screen and detection, waiting out LOADING.

    Stability means the state and the sorted set of interactive match keys
    are identical on two consecutive polls.  On timeout one
    ``SETTLE_TIMEOUT`` event is logged and the last pair is returned.

    Returns:
        tuple[Screen, Detection]: The single screen returned to the caller.
    """
    deadline = self.now() + self.config.settle_timeout
    stable = None
    last = None
    while True:
      screen = self.adb.screenshot()
      detection = self.detector.detect(screen)
      last = (screen, detection)
      if detection.state != GameState.LOADING:
        signature = (detection.state, tuple(sorted(
            (match.name, match.rect.x // 16, match.rect.y // 16)
            for match in detection.matches)))
        if signature == stable:
          return last
        stable = signature
      if self.now() >= deadline:
        self.db.log_event(self.session_id, self.now(), 'SETTLE_TIMEOUT',
                          detection.state.value, {})
        return last
      time.sleep(self.config.poll_delay)

  def run(self):
    """Run the bot until a goal or stop condition is reached.

    Returns:
        SessionStats: Statistics with the terminal ``result`` populated.
    """
    result = 'INTERRUPTED'
    dry_run = self.config.dry_run
    config_json = json.dumps(asdict(self.config), default=_json_default,
                             sort_keys=True)
    stats = SessionStats(started_at=self.now())
    self._stats = stats
    self.counts = {}
    self._pending_gift_before = None
    self._gift_collect_tapped = False
    self._step = None
    self._step_remaining = 0
    session_id = -1
    try:
      session_id = self.db.open_session(config_json, 'INTERRUPTED')
      self.session_id = session_id
      if not self.adb.is_connected():
        result = 'DISCONNECTED'
        return stats
      if not dry_run:
        self.adb.launch(self.config.package)
      while True:
        screen, detection = self.settle()
        stats.screenshots += 1
        state = detection.state
        stats.states_seen[state.value] = (
            stats.states_seen.get(state.value, 0) + 1)
        screen_hash = hashlib.sha256(
            screen.pixels.tobytes()).hexdigest()[:16]
        self.db.log_observation(session_id, self.now(), state.value,
                                detection.matches, self.counts, screen_hash)
        gift_action_taken = self._gift_collect_tapped
        # Phase 1: read coins before tapping the gift icon.
        if (not dry_run and not self._gift_collect_tapped
                and state != GameState.GIFT_DIALOG
                and self.gift.is_ready(screen)):
          before = self.gift.read_coins(screen)
          if before is not None:
            self._pending_gift_before = before
            action = self.gift.plan(screen)
            if action is not None:
              self.db.log_action(
                  session_id, self.now(), action,
                  self._record_outcome(self._execute(action)))
              gift_action_taken = True
          else:
            self._pending_gift_before = None
            self._gift_collect_tapped = False
            failure = ActionResult(
                Action(ActionKind.TAP, None, None, 'gift_coins_unreadable'),
                False, 'coins OCR returned None')
            self.db.log_action(session_id, self.now(), failure.action,
                               self._record_outcome(failure))
            gift_action_taken = True
        # Phase 2: tap Collect only when a before value is pending.
        elif (not dry_run and state == GameState.GIFT_DIALOG
              and not self._gift_collect_tapped):
          if self._pending_gift_before is None:
            before = self.gift.read_coins(screen)
            if before is None:
              failure = ActionResult(
                  Action(ActionKind.TAP, None, None, 'gift_no_before'),
                  False, 'coins OCR returned None on GIFT_DIALOG')
              self.db.log_action(session_id, self.now(), failure.action,
                                 self._record_outcome(failure))
              close = self.navigator.action_for_template(detection,
                                                          'close_dialog')
              if close is not None:
                self.db.log_action(
                    session_id, self.now(), close,
                    self._record_outcome(self._execute(close)))
              gift_action_taken = True
            else:
              self._pending_gift_before = before
          if (self._pending_gift_before is not None
                  and not gift_action_taken):
            action = self.navigator.choose(detection, self.memory, None)
            if (action.kind is ActionKind.TAP
                    and action.reason == 'gift_collect'):
              self.db.log_action(
                  session_id, self.now(), action,
                  self._record_outcome(self._execute(action)))
              self._gift_collect_tapped = True
              gift_action_taken = True
        # Phase 3: verify on a later tick after the dialog closed.
        if (not dry_run and self._gift_collect_tapped
                and self._pending_gift_before is not None
                and state != GameState.GIFT_DIALOG):
          before = self._pending_gift_before
          after = self.gift.read_coins(screen)
          verdict = self.gift.verify(before, after)
          if verdict.ok:
            self.db.log_gift(session_id, self.now(), before, after)
            stats.gifts_collected += 1
          else:
            failure = ActionResult(
                Action(ActionKind.TAP, None, None, 'gift_verify_failed'),
                False, verdict.reason)
            self.db.log_action(session_id, self.now(), failure.action,
                               self._record_outcome(failure))
          self._pending_gift_before = None
          self._gift_collect_tapped = False
        # Inventory and goal.
        if state == GameState.BAG_DIALOG:
          self.counts = self.inventory.read(screen)
          for item in TARGET_IDS:
            count = self.counts.get(item, 0)
            self.db.log_goal_progress(session_id, self.now(), item, count,
                                      self.targets.items[item],
                                      count >= self.targets.items[item])
        if goal_met(self.counts, self.targets):
          result = 'GOAL_MET'
          break
        # Periodic Bag visit.
        if (not gift_action_taken
                and state in (GameState.HOME, GameState.CRAFTING)
                and stats.actions > 0
                and stats.actions % self.config.bag_visit_interval == 0):
          revisit = self.navigator.action_for_template(detection,
                                                       'bag_marker')
          if revisit is not None:
            self.db.log_action(session_id, self.now(), revisit,
                               self._record_outcome(self._execute(revisit)))
            gift_action_taken = True
        # General action (never on a gift-handled or pending-collect tick).
        if not gift_action_taken:
          collect_action = self.navigator.action_for_template(
              detection, 'craft_ready_check')
          if collect_action is not None:
            # completed craft tile: collect; do NOT consume a schedule step
            self.db.log_action(
                session_id, self.now(), collect_action,
                self._record_outcome(self._execute(collect_action)))
          else:
            if self._step_remaining <= 0:
              step = self.planner.next_step()   # PEEK, not advance
              if step is not None:
                self._step = step
                self._step_remaining = step.batch
            craft_action = None
            if self._step is not None and self._step_remaining > 0:
              craft_action = self.navigator.action_for_template(
                  detection, self._step.template)
            if craft_action is not None:
              outcome = self._record_outcome(self._execute(craft_action))
              self.db.log_action(session_id, self.now(), craft_action,
                                 outcome)
              if outcome.ok:
                self._step_remaining -= 1
                if self._step_remaining <= 0:
                  self.planner.mark_placed(self._step.item)
                  self._step = None
            else:
              # UNIVERSAL FALLBACK: dismiss popup / close bag / navigate
              # whatever the current state is when no craft was possible.
              action = self.navigator.choose(detection, self.memory, None)
              self.db.log_action(session_id, self.now(), action,
                                 self._record_outcome(self._execute(action)))
        # Memory transition.
        fresh = [self.memory.mark_visited(
            m.name, (m.rect.x // 16, m.rect.y // 16))
            for m in detection.matches]
        if any(fresh) or state != self.memory.last_state:
          self.memory.reset_progress()
        else:
          self.memory.no_progress += 1
        self.memory.last_state = state
        # Stops.
        if stats.actions >= self.config.max_actions:
          result = 'MAX_ACTIONS'
          break
        if self.memory.no_progress >= self.config.no_progress_limit:
          result = 'NO_PROGRESS'
          break
        if stats.errors >= self.config.error_limit:
          result = 'ERROR_LIMIT'
          break
        if not self.adb.is_connected():
          result = 'DISCONNECTED'
          break
        if self.now() - stats.started_at >= self.config.duration_seconds:
          result = 'DURATION'
          break
    except (KeyboardInterrupt, SystemExit):
      result = 'INTERRUPTED'
    except (AdbError, VisionError, OSError) as exc:
      stats.errors += 1
      result = 'ERROR_LIMIT'
      try:
        self.db.log_event(session_id, self.now(), 'RUN_FAULT', 'UNKNOWN',
                          {'detail': str(exc)})
      except DbError:
        pass
    except DbError as exc:
      stats.errors += 1
      result = 'ERROR_LIMIT'
      try:
        self.db.log_event(session_id, self.now(), 'RUN_FAULT', 'UNKNOWN',
                          {'detail': str(exc)})
      except DbError:
        pass
    finally:
      stats.finished_at = self.now()
      stats.result = result
      try:
        if session_id >= 0:
          self.db.close_session(session_id, result)
      finally:
        try:
          self.db.close()
        except DbError:
          pass
    return stats

  def _execute(self, action):
    """Execute an action, returning an outcome and never raising.

    In dry-run mode no device call is made.  In live mode ADB, vision and OS
    failures are converted into an unsuccessful ``ActionResult``.

    Args:
        action: Action to execute.

    Returns:
        ActionResult: Outcome of the action.
    """
    if self.config.dry_run:
      return ActionResult(action, True, 'dry_run simulated')
    try:
      kind = action.kind
      if kind is ActionKind.TAP and action.point is not None:
        self.adb.tap(action.point)
      elif (kind is ActionKind.SWIPE and action.point is not None
            and action.end is not None):
        self.adb.swipe(action.point, action.end)
      elif kind is ActionKind.BACK:
        self.adb.back()
      elif kind is ActionKind.WAIT:
        time.sleep(self.config.poll_delay)
      return ActionResult(action, True, 'ok')
    except (AdbError, VisionError, OSError) as exc:
      return ActionResult(action, False, str(exc))

  def _record_outcome(self, result):
    """Count an action and forward failures to failure accounting.

    Args:
        result: Outcome to record.

    Returns:
        ActionResult: The unchanged outcome.
    """
    self._stats.actions += 1
    if not result.ok:
      self._count_failure(self.session_id, self.now(), result)
    return result

  def _count_failure(self, session_id, ts, result):
    """Increment the error count and log an ``ACTION_FAILURE`` event.

    Args:
        session_id: Session to attribute the failure to.
        ts: Failure timestamp.
        result: Failed action outcome carrying the detail text.
    """
    self._stats.errors += 1
    self.db.log_event(session_id, ts, 'ACTION_FAILURE', 'UNKNOWN',
                      {'detail': result.detail})
