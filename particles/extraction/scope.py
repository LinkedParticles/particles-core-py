"""Document-scope classification contract.

The general extractor classifies each candidate particle's *scope* — whether
it asserts something about the world (``WORLD``) or about the source
document's own structure / editorial apparatus (``DOCUMENT_META``). This
module holds the small, dependency-free contract shared by the producer
(``particles.extraction.general``) and the three consumers that must keep
document-meta particles out of the factual surface:

* ``particles.ingest.pipeline`` — §6.6 conflict resolution skips them.
* ``particles.operations.lint.contradictions`` — contradiction-checking
  skips them.
* ``particles.operations.query.main`` — the default result set excludes
  them (overridable via ``QueryRequest.include_document_meta``).

The signal lives on a particle's Extension-side ``properties`` dict
, so Core modules never branch on it; only the operation layer and
the extraction pipeline do. ``confidence`` is never used as the scope lever —
a ``DOCUMENT_META`` claim may be perfectly true (§Decision 3).

**The exclusion is lifted per source**. On a rules document
— ``AGENTS.md``, ``CLAUDE.md``, a runbook — the question has no
clean answer, because such a document's *subject matter* is the project's own
apparatus. Measured on the live store (2026-07-25), the classifier hides 178
of a rule file's 1,149 claims, and they are overwhelmingly claims about the
codebase ("the ``particles/store/`` package owns the SQLAlchemy ORM") rather
than about the document. An entry whose tags intersect
``extraction_scope.exempt_source_tags`` therefore stamps
``extraction:scope_action = source_exempt`` on its flagged candidates, and this predicate
lets them through. Scope is still classified and still recorded — only the
behavioural exclusion is lifted.
"""

from __future__ import annotations

from collections.abc import Iterable

# Key on the ``properties`` dict carrying the scope classification. The
# ``extraction:`` prefix is the requirement, taken up;
# particles minted before 1.111.0 carry the bare ``scope`` / ``scope_action``
# spelling, rewritten in place by Alembic 035 and normalised on interchange
# import, so consumers only ever see the prefixed form.
SCOPE_KEY = "extraction:scope"
# The one scope value that triggers exclusion. Absence of the key ⇒ ``WORLD``.
SCOPE_DOCUMENT_META = "DOCUMENT_META"
# Key recording the policy applied to a flagged candidate at extraction time.
SCOPE_ACTION_KEY = "extraction:scope_action"
# ``passthrough`` mode records the classification but applies no exclusion, so
# consumers must treat the particle as ordinary. Its absence ⇒ act (exclude).
SCOPE_ACTION_OBSERVE = "observe"
# the source's genre exempts it from the exclusion. A distinct value
# from ``observe`` on purpose — ``observe`` means "the operator put the whole
# classifier in passthrough mode", and conflating a store-wide evaluation
# posture with a per-source policy would make both unreadable in an audit.
SCOPE_ACTION_SOURCE_EXEMPT = "source_exempt"

# The two actions that record a classification without acting on it.
_NON_EXCLUDING_ACTIONS = frozenset({SCOPE_ACTION_OBSERVE, SCOPE_ACTION_SOURCE_EXEMPT})


def is_excluded_document_meta(properties: dict[str, object] | None) -> bool:
    """Return True if a particle is a document-meta claim that must be hidden.

    True only when the particle was tagged ``extraction:scope == DOCUMENT_META`` **and**
    the recorded scope action neither put the classifier in ``passthrough``
    (``observe``) nor exempted the source's genre (``source_exempt``). This is the single predicate the pipeline, lint, and query
    consumers share — keeping the key/value strings in one place.
    """
    if not properties:
        return False
    return (
        properties.get(SCOPE_KEY) == SCOPE_DOCUMENT_META
        and properties.get(SCOPE_ACTION_KEY) not in _NON_EXCLUDING_ACTIONS
    )


def is_scope_exempt_source(tags: Iterable[str] | None) -> bool:
    """Return True if a corpus entry's tags exempt its claims.

    Membership is ``extraction_scope.exempt_source_tags`` — ``["rule-file"]``
    by default, which is the tag already put on every rule-source entry
    , so ``particles rules sync`` enrols a document with no extra
    gesture.
    An operator extends the exemption to their own genre by adding a tag, and
    disables it entirely by emptying the list.

    Config is read at call time (never captured at import), so
    ``reset_config()`` and runtime reloads take effect.
    """
    if not tags:
        return False
    from particles.config import get_config

    exempt = set(get_config().extraction_scope.exempt_source_tags)
    return bool(exempt) and not exempt.isdisjoint(tags)


def apply_source_exemption(properties: dict[str, object] | None) -> dict[str, object] | None:
    """Stamp the exemption on one candidate's ``properties``.

    Returns the properties to use. A no-op for anything the exclusion would
    not have touched — a ``WORLD`` candidate carries no ``extraction:scope`` key at all
    (labels only what it flags), so there is nothing to exempt, and
    stamping it anyway would write the key onto ~85% of a rule file's output
    for no behavioural difference.

    An existing ``extraction:scope_action`` is left alone: ``observe`` records that the
    operator put the classifier in ``passthrough`` mode, and that fact should
    survive rather than be overwritten by a per-source policy that happens to
    reach the same outcome.
    """
    if not properties or properties.get(SCOPE_KEY) != SCOPE_DOCUMENT_META:
        return properties
    if properties.get(SCOPE_ACTION_KEY) is not None:
        return properties
    stamped = dict(properties)
    stamped[SCOPE_ACTION_KEY] = SCOPE_ACTION_SOURCE_EXEMPT
    return stamped
