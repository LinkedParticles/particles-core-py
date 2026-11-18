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

**The restatement ladder** (the duplicate question of a re-extracted
``MUTABLE`` source): the local ``duplicate_key`` rung (``SAME`` under
the §6.10 key), then the ``duplicate_judge`` model rung, whose
``SAME`` carries nothing until it is admitted. Curation asks the judge alone,
after its own content-hash pass, and the judge counts its verdicts under the
same rung stamp (``ingest/duplicate_judge``).

**The curation duplicate rung**: ``identifier_cross`` (pure),
``DIFFERENT`` when each claim names an identifier the other lacks. Curation's
duplicate step asks it of every candidate pair before the equivalence judge
and withholds the pairs it calls different; it never answers ``SAME``.

**The candidate ladder** (the abstraction gate):
``population_quantifier`` (pure, ``NOT_ENTAILED`` on a kind-ranging
quantifier) and the ``entailment_judge`` model rung.

**The episode ladder** (the one per-claim question):
``credit`` (pure, admitted: a used or directly asserted claim is
``STANDING``), the ``session-anchor`` and ``shape`` shadow rungs (pure,
recorded and never returned), then the stored model reading, one rung per
writer stamp (``prompt.episode@<digest>``, ``regen.episode@<digest>``,
``operator@1``). The reading's cost was paid when it was written; the rung
reads the stored value. Its verdict chooses how pair questions are put to the
owner and is a premise of no other ladder.

Pure: no I/O, no logging.
"""

from __future__ import annotations

from typing import NamedTuple

from particles.core.conflict_resolution import is_generic_instance_pair
from particles.core.duplicate_key import normalize_content
from particles.core.episode import SHAPES_DIGEST, EpisodeInput, EpisodeReading, shapes
from particles.core.generics import population_quantifier
from particles.core.identifier_cross import crossed
from particles.core.rung_ladder import (
    CandidateVerdict,
    ContradictionVerdict,
    CostClass,
    DuplicateVerdict,
    EpisodeVerdict,
    Ladder,
    QuestionKind,
    Rung,
    UpdateVerdict,
)
from particles.core.schema import Particle, is_truth_apt

__all__ = [
    "ADJUDICABILITY",
    "CREDIT",
    "DUPLICATE_KEY",
    "IDENTIFIER_CROSS",
    "RESTATEMENT_KEY",
    "GENERIC_GUARD",
    "SESSION_ANCHOR",
    "SHAPE",
    "POPULATION_QUANTIFIER",
    "CandidateInput",
    "ClaimPair",
    "NormalizedForms",
    "TextPair",
    "UpdatePair",
    "WITHHELD_REASONS",
    "candidate_ladder",
    "contradiction_ladder",
    "duplicate_key_rung",
    "episode_ladder",
    "restatement_ladder",
    "update_ladder",
]

#: A pair of stored claims, as the census holds them.
ClaimPair = tuple[Particle, Particle]
#: Two claim texts, earlier first where the question is directional.
TextPair = tuple[str, str]
#: A candidate abstraction's text and its premises' texts.
CandidateInput = tuple[str, tuple[str, ...]]


class UpdatePair(NamedTuple):
    """The update question's input: two claim texts, earlier first.

    ``sourced`` says whether the shell holds each claim's source passage, which
    decides the confirmation rung: the passage reading when it does, the
    claim-text reading (the assertion door, whose lineage is no source) when
    it does not. The passages themselves stay in the shell.
    """

    earlier: str
    later: str
    sourced: bool = False


def _sourced(pair: UpdatePair) -> bool:
    return pair.sourced


def _unsourced(pair: UpdatePair) -> bool:
    return not pair.sourced


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

    # Version 2: the key folds typographic variants, so its verdicts
    # are a new record set in the probe ledger rather than a continuation of 1's.
    return Rung(name="duplicate_key", version="2", cost=CostClass.LOCAL, decide=decide)


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


def update_ladder(version: str) -> Ladder[UpdatePair, UpdateVerdict]:
    """The update probe as a ladder with a confirmation rung.

    The first reading answers; a first ``CHANGES`` is read again on
    ``verification`` and that reading's verdict, abstention included, replaces
    it, so rung 2.5 retires only on two ``CHANGES``. The confirmation reads the
    source passages when the pair has them (``update_slot_passage``) and the
    two claim texts when it has none (``update_slot_second``, the assertion
    door): one question, more evidence, never two confirmations.
    """
    changes = frozenset({UpdateVerdict.CHANGES.value})
    return Ladder(
        kind=QuestionKind.UPDATE,
        abstain=UpdateVerdict.ABSTAIN,
        rungs=(
            Rung(name="update_slot", version=version, cost=CostClass.MODEL),
            Rung(
                name="update_slot_passage",
                version=version,
                cost=CostClass.MODEL,
                applies=_sourced,
                confirms=changes,
                content_keyed=False,
            ),
            Rung(
                name="update_slot_second",
                version=version,
                cost=CostClass.MODEL,
                applies=_unsourced,
                confirms=changes,
            ),
        ),
    )


def _same_key(pair: TextPair) -> DuplicateVerdict:
    a, b = pair
    same = normalize_content(a) == normalize_content(b)
    return DuplicateVerdict.SAME if same else DuplicateVerdict.ABSTAIN


#: ``SAME`` when the two texts are one sentence under the §6.10 key, whatever
#: their subjects: own predicate, admitted from the start.
RESTATEMENT_KEY: Rung[TextPair, DuplicateVerdict] = Rung(
    name="duplicate_key", version="2", cost=CostClass.LOCAL, decide=_same_key
)


def _identifier_cross(pair: TextPair) -> DuplicateVerdict:
    a, b = pair
    return DuplicateVerdict.DIFFERENT if crossed(a, b) else DuplicateVerdict.ABSTAIN


#: ``DIFFERENT`` when each claim names a digit-bearing identifier the other
#: lacks, read at minimum token length 3. Admitted from the start:
#: on 2026-11-08 the owner labelled 30 randomly drawn pairs it calls
#: ``DIFFERENT`` on their store, and all 30 were correct. A
#: different length, tokenizer or normalisation is a new version, which runs
#: in shadow until it meets the bar on its own labels.
IDENTIFIER_CROSS: Rung[TextPair, DuplicateVerdict] = Rung(
    name="identifier_cross", version="1", cost=CostClass.PURE, decide=_identifier_cross
)

#: Why each withholding rung keeps a duplicate candidate pair off the curation
#: queue, by rung name, as every surface that discloses the count words it.
WITHHELD_REASONS: dict[str, str] = {
    IDENTIFIER_CROSS.name: "each claim names an identifier the other lacks",
}


def restatement_ladder(judge_version: str) -> Ladder[TextPair, DuplicateVerdict]:
    """The duplicate ladder a re-extracted ``MUTABLE`` source asks of each pair.

    The key rung, then the equivalence judge. The judge is a model
    rung, which is never run in shadow, so its admission period is
    held at the write instead (``generation.restatement.judge_admitted``): until
    the owner admits the judge, its ``SAME`` is recorded and counted and carries
    nothing.
    """
    return Ladder(
        kind=QuestionKind.DUPLICATE,
        abstain=DuplicateVerdict.ABSTAIN,
        rungs=(
            RESTATEMENT_KEY,
            Rung(name="duplicate_judge", version=judge_version, cost=CostClass.MODEL),
        ),
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


# ---------------------------------------------------------------------------
# The episode ladder
# ---------------------------------------------------------------------------


def _credit(item: EpisodeInput) -> EpisodeVerdict:
    if item.credited or item.asserted:
        return EpisodeVerdict.STANDING
    return EpisodeVerdict.ABSTAIN


#: A claim someone acted on or asserted is never folded. Its
#: ``STANDING`` leaves the claim's cards as they are, so it is admitted.
CREDIT: Rung[EpisodeInput, EpisodeVerdict] = Rung(
    name="credit",
    version="1",
    cost=CostClass.PURE,
    decide=_credit,
    content_keyed=False,
)


def _session_anchor(item: EpisodeInput) -> EpisodeVerdict:
    if item.valid_from_basis == "session":
        return EpisodeVerdict.SESSION_BOUND
    return EpisodeVerdict.ABSTAIN


#: An act the extractor anchored at the session's date. Shadow:
#: it reaches acts only, and only claims from extractor 0.18.0 onward.
SESSION_ANCHOR: Rung[EpisodeInput, EpisodeVerdict] = Rung(
    name="session-anchor",
    version="1",
    cost=CostClass.PURE,
    decide=_session_anchor,
    shadow=True,
    content_keyed=False,
)


def _shape(item: EpisodeInput) -> EpisodeVerdict:
    if shapes(item.content):
        return EpisodeVerdict.SESSION_BOUND
    return EpisodeVerdict.ABSTAIN


#: The lexical session-report shapes. Shadow: about 0.73 precise on the
#: author's sample (Context).
SHAPE: Rung[EpisodeInput, EpisodeVerdict] = Rung(
    name="shape",
    version=SHAPES_DIGEST,
    cost=CostClass.PURE,
    decide=_shape,
    shadow=True,
)


def _stored_reading(item: EpisodeInput) -> EpisodeVerdict:
    return item.reading.verdict if item.reading is not None else EpisodeVerdict.ABSTAIN


def episode_ladder(reading: EpisodeReading | None) -> Ladder[EpisodeInput, EpisodeVerdict]:
    """The episode ladder for one claim, its last rung named by the stored reading's stamp.

    A reading is a rung per writer stamp, so the ladder is
    built per claim: with no reading, the claim settles on ``credit`` or
    abstains, and the shadow rungs' verdicts are still recorded.
    """
    rungs: list[Rung[EpisodeInput, EpisodeVerdict]] = [CREDIT, SESSION_ANCHOR, SHAPE]
    if reading is not None:
        stamp = reading.stamp
        rungs.append(
            Rung(
                name=stamp.name,
                version=stamp.version,
                cost=CostClass.MODEL,
                decide=_stored_reading,
                content_keyed=False,
            )
        )
    return Ladder(kind=QuestionKind.EPISODE, abstain=EpisodeVerdict.ABSTAIN, rungs=tuple(rungs))
