#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports
import sqlite3
#    script imports
# imports


# constants
# constants


# classes
class SQLiteConnector:
  '''SQLite Context Manager'''
  def __init__(self, db_file: str):
    self.file_name = db_file
    self.conn = sqlite3.connect(self.file_name)

  def __enter__(self):
    return self.conn.cursor()

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.conn.commit()
    self.conn.close()


class DBFeeder:
  '''SQLite database feeder for all simulations'''

  def __init__(self):
    self._intialize_db()

  def _intialize_db(self):
    pass

# classes


# functions
def function_name():
  pass
# functions


# main
def main():
  pass


# if main script
if __name__ == '__main__':
  main()
