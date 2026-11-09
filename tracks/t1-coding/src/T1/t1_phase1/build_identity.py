"""Unfrozen source checkout; the offline builder replaces this in its context only.

A direct Dockerfile build carries this explicit unknown identity. It must not be
presented as a frozen build merely because it has a candidate profile label.
"""
IDENTITY = {'source_state': 'unfrozen', 'source_commit': None, 'context_sha256': None}
