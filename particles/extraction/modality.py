# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The adjudicability-default classifiers and their identities.

Two producers write ``assertion_modality`` with a classifier: the extractors
that classify on their extraction call (the general extractor's
``prompt.modality`` component, and the journal extractor's rules
component), and the standalone regeneration call here, which runs the general
extractor's component over one stored claim. A classifier's identity is its
extraction component's ``name@digest``, so a rule edit is visible as a new
identity and an unchanged rule keeps one identity whichever producer ran it.

Client layer: no store, corpus, db or ingest import.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from particles.core.modality import classifier_identity
from particles.core.schema import AssertionModality
from particles.extraction import general, journal

log = logging.getLogger(__name__)

#: Component names whose text carries a modality rule. The minting snapshot's
#: component record names one of these when its extraction classified.
MODALITY_COMPONENTS: frozenset[str] = frozenset(
    {general.PROMPT_MODALITY, journal.PROMPT_JOURNAL_RULES}
)

#: Extractors that classify modality at all. A claim one of these produced with
#: no component record is a pre-stamp classification; any other extractor's
#: claim is adjudicable by construction.
CLASSIFYING_EXTRACTORS: frozenset[str] = frozenset({general.EXTRACTOR_ID, journal.EXTRACTOR_ID})


def regenerable(*, extractor_name: str | None, classifier: str) -> bool:
    """Whether the standalone classifier may reclassify a claim.

    The standalone classifier runs the general extractor's rule over one claim
    out of context. A journal claim was classified by the journal prompt, whose
    modality rule is part of its whole rules block and carries guidance the
    general rule lacks (an opinion is never left ``FALSIFIABLE``), so running
    the general rule over it would replace a better-informed verdict with a
    worse one. Journal claims are reclassified by re-extraction instead.
    """
    if extractor_name == journal.EXTRACTOR_ID:
        return False
    return not classifier.startswith(f"{journal.PROMPT_JOURNAL_RULES}@")


def current_modality_classifiers() -> frozenset[str]:
    """The classifier identities in force under the code as it stands.

    Read at call time, never captured, so an edited rule (or a test patching
    one) is seen at once.
    """
    general_component = general.modality_prompt_component()
    identities = {classifier_identity(general_component.name, general_component.digest)}
    for component in journal.journal_prompt_components():
        if component.name == journal.PROMPT_JOURNAL_RULES:
            identities.add(classifier_identity(component.name, component.digest))
    return frozenset(identities)


def regeneration_classifier() -> str:
    """The identity a regenerated claim is stamped with: the general rule's."""
    component = general.modality_prompt_component()
    return classifier_identity(component.name, component.digest)


_CLASSIFY_FRAME = """You classify ONE claim, given in the fenced block, by the
kind of assertion it makes. Read the claim only; do not judge whether it is
true. Apply this rule:
"""

_CLASSIFY_OUTPUT = """

Reply with one JSON object and nothing else:
{"assertion_modality": "FALSIFIABLE", "EVALUATIVE", "EXPERIENTIAL", or "CONSTITUTIVE"}"""


@dataclass(frozen=True)
class ModalityVerdict:
    """One standalone classification: the value and the pairing that served it."""

    modality: AssertionModality
    provider_model: str
    classifier: str


def parse_modality_reply(raw: str) -> AssertionModality | None:
    """The modality a reply names, or ``None`` when it names no valid one.

    Unlike the extraction parser this never falls back to ``FALSIFIABLE``: a
    stored claim already has a value, and writing a fallback over it would
    record a classification that never happened.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("assertion_modality")
    if not isinstance(value, str):
        return None
    try:
        return AssertionModality(value.strip().upper())
    except ValueError:
        return None


async def classify_modality(content: str) -> ModalityVerdict | None:
    """Classify one stored claim's adjudicability default.

    One completion on the ``extraction`` purpose, the general extractor's own
    rule component in a one-claim frame. The claim came from an untrusted
    source, so it is fenced in the user turn as the structurizer fences its
    input. Every failure mode returns ``None`` and leaves the claim as it was.
    """
    from particles.llm import complete_with_provider_model, fenced_prompt

    component = general.modality_prompt_component()
    instructions = _CLASSIFY_FRAME + component.rule + _CLASSIFY_OUTPUT
    system, user = fenced_prompt(instructions, content, label="claim")
    try:
        raw, provider_model = await complete_with_provider_model(
            "extraction", user, max_tokens=64, system=system
        )
    except Exception as exc:
        log.warning("Modality classifier call failed: %s", exc)
        return None
    modality = parse_modality_reply(raw)
    if modality is None:
        return None
    return ModalityVerdict(
        modality=modality,
        provider_model=provider_model,
        classifier=classifier_identity(component.name, component.digest),
    )
