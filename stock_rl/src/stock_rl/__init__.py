#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''NIFTY-RL backtest core.

Pure standard library by design. Every quantity a backtest needs (returns,
drawdown, turnover, the normal CDF used by the Deflated Sharpe Ratio) is
available in the standard library, so the core carries zero runtime
dependencies. See ``engine.py`` for the bar-count ceiling that motivates
that choice and the upgrade path past it.
'''

__version__ = '0.1.0'

__all__ = ['__version__']
