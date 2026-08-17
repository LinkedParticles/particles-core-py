"""Stance-particle helpers.

A *stance* is an ordinary ``FALSIFIABLE`` :class:`~particles.core.schema.Particle`
asserting an attribution fact — *"agent A endorses / disputes claim B"* — bound
to its target by an outbound ``ENDORSES`` / ``DISPUTES`` relation. The edge is
the role marker. Two ``stance:``-prefixed ``properties`` keys
 carry the holder identity and the optional attitude
magnitude.

Core stays I/O-free: these are pure helpers over a particle's
``properties``. The authoritative edge-based "is this a stance" check (the role
marker per §1) lives in the store / operations layer where a session is
available; the property marker below is the cheap, co-stamped signal the
extractor writes *alongside* the edge, used to keep stances out of factual
top-k (§6) without an extra edge query per candidate.
"""

from __future__ import annotations

from particles.core.schema import Particle, RelationType

#: ``properties`` key carrying the stance holder's ``platform:identifier`` (§3).
STANCE_HOLDER_KEY = "stance:holder"

#: ``properties`` key carrying the optional attitude strength, float in [0, 1] (§3).
STANCE_MAGNITUDE_KEY = "stance:magnitude"

#: The two relation kinds whose outbound edge marks a particle as a stance
#:. Both asymmetric (stance → target); neither is in
#: ``relation_store._SYMMETRIC_KINDS``.
STANCE_KINDS: frozenset[RelationType] = frozenset({RelationType.ENDORSES, RelationType.DISPUTES})


def holder_from_properties(properties: dict[str, object] | None) -> str | None:
    """Return the ``stance:holder`` identifier from a raw ``properties`` dict.

    Works on a bare ``properties`` mapping (e.g. an extraction
    ``CandidateParticle``) before a :class:`Particle` exists.
    """
    if not properties:
        return None
    value = properties.get(STANCE_HOLDER_KEY)
    return value if isinstance(value, str) and value else None


def stance_holder(particle: Particle) -> str | None:
    """Return the ``stance:holder`` identifier, or ``None`` if absent / not a stance."""
    return holder_from_properties(particle.properties)


def stance_magnitude(particle: Particle) -> float | None:
    """Return the ``stance:magnitude`` float in [0, 1], or ``None`` if unqualified.

    ``None`` (absent) means the source expressed the attitude without a strength
    qualifier — distinct from an explicit ``0.0``.
    """
    props = particle.properties
    if not props:
        return None
    value = props.get(STANCE_MAGNITUDE_KEY)
    # bool is an int subclass; exclude it explicitly.
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def has_stance_marker(particle: Particle) -> bool:
    """Cheap, I/O-free predicate: does this particle carry the ``stance:holder`` marker?

    Stance particles are stamped with ``stance:holder`` alongside their outbound
    ``ENDORSES`` / ``DISPUTES`` edge, so this is the co-stamped
    signal for keeping stances out of factual top-k (§6) and out of §6.6
    candidacy without a per-candidate edge query. The authoritative role marker
    is the edge itself (§1); use the store-level edge check when correctness
    against hand-edited data matters.
    """
    return stance_holder(particle) is not None
