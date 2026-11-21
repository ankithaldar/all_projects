#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Immutable data containers and enumerations shared by cat_game_bot."""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

__all__ = (
  'Point', 'Rect', 'Screen', 'Match', 'Detection', 'GameState',
  'ActionKind', 'Action', 'ActionResult', 'GiftResult', 'InventoryItem',
  'CraftStep', 'SessionStats',
)


@dataclass(frozen=True)
class Point:
  """An immutable integer coordinate in screen space."""

  x: int
  y: int


@dataclass(frozen=True)
class Rect:
  """An immutable axis-aligned rectangle with a half-open extent."""

  x: int
  y: int
  width: int
  height: int

  def center(self) -> Point:
    """Return the integer midpoint of this rectangle.

    Returns:
      The floor-division midpoint point.
    """
    return Point(self.x + self.width // 2, self.y + self.height // 2)

  def contains(self, point: Point) -> bool:
    """Report whether a point falls inside the half-open rectangle.

    Args:
      point: The point to test.

    Returns:
      True when the point's coordinates are within the rectangle.
    """
    return (self.x <= point.x < self.x + self.width
            and self.y <= point.y < self.y + self.height)

  def area(self) -> int:
    """Return the rectangle area in pixels.

    Returns:
      The product of width and height.
    """
    return self.width * self.height


@dataclass(eq=False)
class Screen:
  """A captured screen frame backed by a pixel array."""

  pixels: np.ndarray

  @property
  def width(self) -> int:
    """Return the frame width in pixels.

    Returns:
      The second dimension of the pixel array.
    """
    return self.pixels.shape[1]

  @property
  def height(self) -> int:
    """Return the frame height in pixels.

    Returns:
      The first dimension of the pixel array.
    """
    return self.pixels.shape[0]


@dataclass(frozen=True)
class Match:
  """A single template match result."""

  name: str
  rect: Rect
  score: float


@dataclass(frozen=True)
class Detection:
  """The recognised game state and its interactive template matches."""

  state: GameState
  matches: tuple[Match, ...]
  timestamp: float


class GameState(Enum):
  """The distinct screens the bot can recognise."""

  UNKNOWN = 'UNKNOWN'
  LOADING = 'LOADING'
  TAP_TO_START = 'TAP_TO_START'
  TUTORIAL = 'TUTORIAL'
  HOME = 'HOME'
  CRAFTING = 'CRAFTING'
  CATALOG = 'CATALOG'
  POPUP = 'POPUP'
  RECIPE_DIALOG = 'RECIPE_DIALOG'
  BAG_DIALOG = 'BAG_DIALOG'
  GIFT_DIALOG = 'GIFT_DIALOG'
  DIALOG = 'DIALOG'


class ActionKind(Enum):
  """The kinds of low-level device action the bot may issue."""

  TAP = 'TAP'
  SWIPE = 'SWIPE'
  WAIT = 'WAIT'
  BACK = 'BACK'
  STOP = 'STOP'


@dataclass(frozen=True)
class Action:
  """An immutable action plan produced by the navigator."""

  kind: ActionKind
  point: Point | None
  end: Point | None
  reason: str


@dataclass(frozen=True)
class ActionResult:
  """The immutable outcome of executing a planned action."""

  action: Action
  ok: bool
  detail: str


@dataclass
class GiftResult:
  """The verdict of a gift coin-delta verification."""

  ok: bool
  delta: int
  reason: str


@dataclass
class InventoryItem:
  """A named bag item with its observed count and icon rectangle."""

  name: str
  count: int
  rect: Rect


@dataclass(frozen=True)
class CraftStep:
  """One immutable crafting recipe to attempt."""

  item: str
  section: str
  template: str
  batch: int
  cost: int


@dataclass
class SessionStats:
  """Mutable counters and the terminal result of a bot session."""

  actions: int = 0
  screenshots: int = 0
  errors: int = 0
  gifts_collected: int = 0
  states_seen: dict[str, int] = field(default_factory=dict)
  started_at: float = 0.0
  finished_at: float = 0.0
  result: str = 'INTERRUPTED'
