#! /usr/bin/env python
"""Get the list of user functions and classes, for generating documentation"""
from inspect import ismodule

import tbplas as tb


# Get initial labels
attr_dict = tb.__dict__
labels = attr_dict.keys()

# Filter builtin names
labels = set([_ for _ in labels if _[0] != "_"])

# Filter Exceptions
labels = labels.difference(dir(tb.exceptions))

# Filter physical constants
labels = labels.difference(dir(tb.constants))

# Echo
labels = sorted(labels)
for label in labels:
    attr = attr_dict[label]
    if not ismodule(attr):
        print(label)
