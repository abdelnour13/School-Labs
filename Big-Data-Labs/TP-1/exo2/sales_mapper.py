#!/usr/bin/env python
"""mapper.py"""

import sys

# input comes from STDIN (standard input)
for line in sys.stdin:

    # remove leading and trailing whitespace
    line = line.strip()

    # split the line into words
    # get rid of the empty strings
    words = list(filter(lambda x : len(x) != 0,line.split('\t')))

    # the store and the price
    store = words[2]
    price = words[-2]

    # write to stdout
    print('%s\t%s' % (store, price))