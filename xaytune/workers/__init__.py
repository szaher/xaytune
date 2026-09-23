"""Workers: the processes that actually train.

A worker receives a compiled config and produces observations. It never sees a
candidate, a runtime, an envelope or a cursor -- it knows how to train and how
to report what happened, and nothing about how either is recorded.
"""
