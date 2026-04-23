from __future__ import annotations


class CoinGenerator:
  COINS_PER_TICK = 210

  def __init__(self, initial_balance: int = 0):
    self._balance = initial_balance

  @property
  def balance(self) -> int:
    return self._balance

  def tick(self) -> None:
    self._balance += self.COINS_PER_TICK

  def spend(self, amount: int) -> bool:
    if amount < 0:
      raise ValueError(f"Cannot spend negative amount: {amount}")
    if self._balance < amount:
      return False
    self._balance -= amount
    return True

  def add(self, amount: int) -> None:
    if amount < 0:
      raise ValueError(f"Cannot add negative amount: {amount}")
    self._balance += amount

  def reset(self, initial: int = 0) -> None:
    self._balance = initial
