"""Machinery for building the manual gold standard.

The clinical judgements are made by a person reading reports; this package only
screens, selects, records and writes them back. Keeping the two apart is what makes
the labels auditable: every abnormal cell in the CSV traces to a quoted sentence in
the evidence store, and the CSV can be rebuilt from that store at any time.
"""
