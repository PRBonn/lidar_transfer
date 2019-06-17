#!/usr/bin/env python3

import sys


def progressbar(name, value, endvalue, bar_length=50):
  percent = float(value) / endvalue
  arrow = '-' * int(round(percent * bar_length) - 1) + '|'
  spaces = ' ' * (bar_length - len(arrow))
  sys.stdout.write("\r")
  sys.stdout.write(" " * 80)
  sys.stdout.write("\r{0} {1}: [{2}]{3}%".format(
                   name, value, arrow + spaces, int(round(percent * 100))))
  sys.stdout.flush()
