# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The engine's ladders: which rungs answer each question kind, in which order.

Each function below declares one :class:`~particles.core.rung_ladder.Ladder`.
The deterministic rungs carry their own ``decide`` and are defined here; the
model rungs are declared with ``decide=None`` and bound in the shell to the
completion they ask. A model rung's version is the prompt hash of the probe
it stands for (:func:`~particles.core.probe_verdict.prompt_hash`), which the
shell passes in, so a prompt edit changes the stamp every verdict carries.

**The contradiction ladder** (the census, L-SEM-01):

* ``adjudicability`` (pure): ``NO`` when either claim is not truth-apt,
  read from the stored modality only.
* ``generic_guard`` (pure): ``NO`` when exactly one claim is generic.
* ``duplicate_key`` (local): ``NO`` when the two claims are the same sentence
  under the §6.10 normalisation, whatever their subject sets and
  holders; the full §6.10 identity is the narrower case of the same test.
  A sentence does not contradict itself, and this case was 170 of the 200
  pairs a capped census probed before the rung existed. The
  census hands the rung a :class:`NormalizedForms` memo so each text is
  normalised once per run, not once per pair.
* ``contradiction_probe`` (model): the census probe.
* ``second_reading`` (model, confirmation on ``YES``): the context-rich
  second reading, whose verdict replaces the probe's.

**The update ladder** (rung 2.5): ``update_slot`` (model,
``semantic_lint``) and ``update_slot_second`` (model, ``verification``,
confirmation on ``CHANGES``).

**The duplicate ladder**: the ``duplicate_judge`` model rung.

**The candidate ladder** (the abstraction gate):
``population_quantifier`` (pure, ``NOT_ENTAILED`` on a kind-ranging
quantifier) and the ``entailment_judge`` model rung.

Pure: no I/O, no logging.
"""

from __future__ import annotations

from particles.core.conflict_resolution import is_generic_instance_pair
from particles.core.duplicate_key import normalize_content
from particles.core.generics import population_quantifier
from particles.core.rung_ladder import (
    CandidateVerdict,
    ContradictionVerdict,
    CostClass,
    DuplicateVerdict,
    Ladder,
    QuestionKind,
    Rung,
    UpdateVerdict,
)
from particles.core.schema import Particle, is_truth_apt

__all__ = [
    "ADJUDICABILITY",
    "DUPLICATE_KEY",
    "GENERIC_GUARD",
    "POPULATION_QUANTIFIER",
    "CandidateInput",
    "ClaimPair",
    "NormalizedForms",
    "TextPair",
    "candidate_ladder",
    "contradiction_ladder",
    "duplicate_key_rung",
    "duplicate_ladder",
    "update_ladder",
]

#: A pair of stored claims, as the census holds them.
ClaimPair = tuple[Particle, Particle]
#: Two claim texts, earlier first where the question is directional.
TextPair = tuple[str, str]
#: A candidate abstraction's text and its premises' texts.
CandidateInput = tuple[str, tuple[str, ...]]


def _adjudicability(pair: ClaimPair) -> ContradictionVerdict:
    a, b = pair
    if is_truth_apt(a) and is_truth_apt(b):
        return ContradictionVerdict.ABSTAIN
    return ContradictionVerdict.NO


def _generic_guard(pair: ClaimPair) -> ContradictionVerdict:
    if is_generic_instance_pair(*pair):
        return ContradictionVerdict.NO
    return ContradictionVerdict.ABSTAIN


class NormalizedForms:
    """Each claim's §6.10 normalised text, computed once per particle id.

    The duplicate-key rung compares two normalised texts, and a census asks it
    of every candidate pair the pure rungs let through: some 2.66 million
    pairs over 55,578 distinct texts on the owner's store, so normalising per
    pair did the same work fifty times over. The shell makes one of these per
    run and hands it to :func:`duplicate_key_rung`; the rung reads a claim's
    form through it and the first read computes it. A memo of a pure function
    over the particle's own text: no I/O, and the same answer whoever asks.
    """

    def __init__(self) -> None:
        self._forms: dict[str, str] = {}

    def of(self, particle: Particle) -> str:
        """``normalize_content`` of the particle's text, computed on first read."""
        form = self._forms.get(particle.id)
        if form is None:
            form = self._forms[particle.id] = normalize_content(particle.content)
        return form

    def __len__(self) -> int:
        return len(self._forms)


def duplicate_key_rung(
    forms: NormalizedForms | None = None,
) -> Rung[ClaimPair, ContradictionVerdict]:
    """The duplicate-key rung, reading normalised texts through ``forms`` when given.

    With no memo each call normalises both texts afresh, which is
    :data:`DUPLICATE_KEY`; the verdict is the same either way.
    """

    def decide(pair: ClaimPair) -> ContradictionVerdict:
        a, b = pair
        if forms is None:
            same = normalize_content(a.content) == normalize_content(b.content)
        else:
            same = forms.of(a) == forms.of(b)
        return ContradictionVerdict.NO if same else ContradictionVerdict.ABSTAIN

    return Rung(name="duplicate_key", version="1", cost=CostClass.LOCAL, decide=decide)


#: Neither claim is a truth-apt claim, so there is no shared truth to contradict.
ADJUDICABILITY: Rung[ClaimPair, ContradictionVerdict] = Rung(
    name="adjudicability",
    version="1",
    cost=CostClass.PURE,
    decide=_adjudicability,
    # Reads each record's stored modality, not its text.
    content_keyed=False,
)
#: A generic against an instance claim is not an adjudicable pair.
GENERIC_GUARD: Rung[ClaimPair, ContradictionVerdict] = Rung(
    name="generic_guard", version="1", cost=CostClass.PURE, decide=_generic_guard
)
#: The same sentence twice is not a contradiction (normalisation).
DUPLICATE_KEY: Rung[ClaimPair, ContradictionVerdict] = duplicate_key_rung()


def contradiction_ladder(
    probe_version: str,
    reading_version: str,
    *,
    forms: NormalizedForms | None = None,
) -> Ladder[ClaimPair, ContradictionVerdict]:
    """The census's contradiction ladder at the given prompt versions.

    ``forms`` is the run's memo of normalised texts for the duplicate-key rung
    (:class:`NormalizedForms`); without one the rung normalises per pair.
    """
    return Ladder(
        kind=QuestionKind.CONTRADICTION,
        abstain=ContradictionVerdict.ABSTAIN,
        rungs=(
            ADJUDICABILITY,
            GENERIC_GUARD,
            DUPLICATE_KEY if forms is None else duplicate_key_rung(forms),
            Rung(name="contradiction_probe", version=probe_version, cost=CostClass.MODEL),
            Rung(
                name="second_reading",
                version=reading_version,
                cost=CostClass.MODEL,
                confirms=frozenset({ContradictionVerdict.YES.value}),
            ),
        ),
    )


def update_ladder(version: str) -> Ladder[TextPair, UpdateVerdict]:
    """The update probe as a two-rung ladder with a confirmation rung.

    The first reading answers; a first ``CHANGES`` is read again on
    ``verification`` and the second reading's verdict, abstention included,
    replaces it, so rung 2.5 retires only on two ``CHANGES``.
    """
    return Ladder(
        kind=QuestionKind.UPDATE,
        abstain=UpdateVerdict.ABSTAIN,
        rungs=(
            Rung(name="update_slot", version=version, cost=CostClass.MODEL),
            Rung(
                name="update_slot_second",
                version=version,
                cost=CostClass.MODEL,
                confirms=frozenset({UpdateVerdict.CHANGES.value}),
            ),
        ),
    )


def duplicate_ladder(version: str) -> Ladder[TextPair, DuplicateVerdict]:
    """The curation duplicate judge as a one-rung ladder."""
    return Ladder(
        kind=QuestionKind.DUPLICATE,
        abstain=DuplicateVerdict.ABSTAIN,
        rungs=(Rung(name="duplicate_judge", version=version, cost=CostClass.MODEL),),
    )


def _population_quantifier(item: CandidateInput) -> CandidateVerdict:
    if population_quantifier(item[0]) is not None:
        return CandidateVerdict.NOT_ENTAILED
    return CandidateVerdict.ABSTAIN


#: A claim ranging over a kind is not entailed by claims about its members.
POPULATION_QUANTIFIER: Rung[CandidateInput, CandidateVerdict] = Rung(
    name="population_quantifier",
    version="1",
    cost=CostClass.PURE,
    decide=_population_quantifier,
)


def candidate_ladder(judge_version: str) -> Ladder[CandidateInput, CandidateVerdict]:
    """The abstraction gate: the deterministic pre-check, then the entailment judge."""
    return Ladder(
        kind=QuestionKind.ENTAILMENT,
        abstain=CandidateVerdict.ABSTAIN,
        rungs=(
            POPULATION_QUANTIFIER,
            Rung(name="entailment_judge", version=judge_version, cost=CostClass.MODEL),
        ),
    )
