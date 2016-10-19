#!/usr/bin/env python3
"""Dripfeed.
Read the input file or stdin and write it out, one line at a time with a delay.

Usage:
  dripfeed.py [options] FILES ...
  dripfeed.py [options] -
  dripfeed.py (-h | --help)
  dripfeed.py --version

Reads FILE file(s) and rewrite it out one line at a time to with a delay of N seconds between lines.
Use - to read from sys.stdin and write to sys.stdout

Options:
  -i N --interval=seconds  Delay in seconds between writing lines [default: 5]
  -h --help                Show this screen.
  --version                Show version.


"""
import os
from docopt import docopt
import sys
import fileinput
import time

if __name__ == '__main__':
    arguments = docopt(__doc__, version='Dripfeed 1.0')
    #print(arguments)

    with fileinput.input(files=arguments['FILES'], inplace=True) as in_file:
        for line in in_file:
            time.sleep(int(arguments['--interval']))
            print(line, flush=True, end='') # line already has newline


