# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The past-value marker: a claim that says its value is over (guard 2).

Rung 2.5 retires an older claim when a newer one gives its slot a new value.
A claim that records only a past value ("worked at <employer> until <year>",
"used to <do something>", "As of <date>, when <a state that is over>, ...")
is not a current value at all, so a later value never replaces it: both stay
true. The update probe misreads some of these as replaced, and the
contradiction probe in front of it used to keep them. the
update probe decides alone, except for a pair whose older claim carries one of
the markers below, which keeps the contradiction probe in front as before.

The list is deliberately narrow. A marker can only keep a claim ACTIVE: a
false hit costs one update that today's gate would also have needed a
contradiction for, while a miss leaves the pair to the update probe alone.
"until" counts only with a year, since a bounded stay in the present tense
("is in <city> until the end of <month>") is a current value a later claim may
replace. Pure: no I/O, regex only, as :mod:`particles.core.generics`.
"""

from __future__ import annotations

import re

__all__ = ["records_past_value"]

_YEAR = r"(?:19|20)\d\d"

_PAST_VALUE_MARKERS: tuple[re.Pattern[str], ...] = (
    # A value said to have ended at a year: "until 2025", "until March 2025".
    re.compile(rf"\b(?i:until)\s+(?:[A-Z][a-z]+\s+)?{_YEAR}\b"),
    # A value placed before or back in a year: "Before 2024 ...", "Back in 2022 ...".
    re.compile(rf"\b(?:before|back in)\s+{_YEAR}\b", re.IGNORECASE),
    # A value said to be over: previously, formerly, used to, no longer, had been.
    re.compile(r"\b(?:previously|formerly|used to|no longer|had been)\b", re.IGNORECASE),
    # A restatement of a time that is over: "As of <date>, when ...".
    re.compile(r"^\s*as of [^,]{1,40},\s*when\b", re.IGNORECASE),
)


def records_past_value(content: str) -> bool:
    """Whether ``content`` says its value is over, by one of the markers above."""
    return any(pattern.search(content) for pattern in _PAST_VALUE_MARKERS)
