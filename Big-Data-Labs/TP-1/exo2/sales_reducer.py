#!/usr/bin/env python
"""reducer.py"""

import sys

current_store = None
current_total = 0
store = None

# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()

    # parse the input we got from mapper.py
    store, price = line.split('\t', 1)
    # convert count (currently a string) to int
    try:
        price = float(price)
    except ValueError:
        # count was not a number, so silently
        # ignore/discard this line
        continue

    # this IF-switch only works because Hadoop sorts map output
    # by key (here: word) before it is passed to the reducer
    if current_store == store:
        current_total += price
    else:
        if current_store:
            # write result to STDOUT
            print('%s\t%s' % (current_store, current_total))
        current_total = price
        current_store = store

# do not forget to output the last word if needed!
if current_store == store:
    print('%s\t%s' % (current_store, current_total))